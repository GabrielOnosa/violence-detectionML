import torch
import torchvision
import os
import cv2
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
from ultralytics import YOLO
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

yolo_pose = YOLO('yolov8s-pose.pt')
class PicturesDatasetEmbeddings(torch.utils.data.Dataset):
    def __init__(self, directory, embeddings_directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.samples = []
        self.embeddings_directory = embeddings_directory
        class_to_idx = {'Fight': 1, 'NonFight': 0}
        embedding_file = pd.read_csv(self.embeddings_directory)
        for folder in os.listdir(self.directory):
            folder_path = os.path.join(self.directory, folder)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                row_from_csv = embedding_file.loc[embedding_file['title'] == filename].iloc[0, :-2].values
                row_from_csv = row_from_csv.astype(np.float32)  # Convert to float
                row_from_csv = torch.from_numpy(row_from_csv)
                element_tuple = (file_path, class_to_idx[folder], row_from_csv)
                self.samples.append(element_tuple)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        element = self.samples[idx]
        vr = VideoReader(element[0], ctx=cpu(0))  # CPU context
        max_people = 4
        num_keypoints = 17
        features_per_keypoint = 3
        fixed_per_frame = max_people * num_keypoints * features_per_keypoint  # 204
        frame_list = [38, 76, 114]
        final_frame = []
        yolo_embeddings = []
        for index in frame_list:
            temp_frame = vr[index]
            temp_frame = temp_frame.asnumpy()
            result = yolo_pose(temp_frame, verbose=False)
            h, w, _ = temp_frame.shape  # original frame shape
            if result[0].keypoints is not None:
                kpts = result[0].keypoints.data.cpu().numpy()
                kpts[..., 0] /= w  # divide all x coordinates by frame width
                kpts[..., 1] /= h  # divide all y coordinates by frame height
                # print(kpts.shape)
                # print(kpts)
                num_people_detected = min(kpts.shape[0], max_people)
                kpts = kpts[:num_people_detected]
                kpts = kpts.reshape(-1)
                if kpts.shape[0] < fixed_per_frame:
                    pad = np.zeros(fixed_per_frame - kpts.shape[0], dtype=np.float32)
                    kpts = np.concatenate([kpts, pad])
            else:
                kpts = np.zeros(fixed_per_frame, dtype=np.float32)
            yolo_embeddings.append(kpts)
            temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
            final_frame.append(temp_frame)
        final_frame = np.array(final_frame)
        final_frame = torch.from_numpy(final_frame)
        final_frame = final_frame.float()
        final_frame = self.transform(final_frame)
        yolo_embeddings = np.array(yolo_embeddings)
        yolo_embeddings = yolo_embeddings.flatten()
        yolo_embeddings = torch.from_numpy(yolo_embeddings)
        return final_frame, element[2], yolo_embeddings, element[1]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
])

if __name__ == '__main__':
    train_dataset = PicturesDatasetEmbeddings(r'D:\Downloads\database\RWF-2000\train', 'embeddings_from_gemini_768_no_keywords.csv', transform = transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle = True)
    val_dataset = PicturesDatasetEmbeddings(r'D:\Downloads\database\RWF-2000\val', 'embeddings_from_gemini_768val_no_keywords.csv', transform = transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)



    class ModifiedResnetwithEmbeddings(torch.nn.Module):
        def __init__(self, resnet, number_features_embedding):
            super().__init__()
            self.up_until_embeddings = torch.nn.Sequential(*list(resnet.children())[:-1])
            print('resnet50 components: \n')
            print(list(resnet.children()))
            print("")
            print("")
            print(list(resnet.children())[-1])
            print("")
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features = resnet.fc.in_features + 768 + 612, out_features = 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1024, 2)  # 2 classes: violence / non-violence
            )
        def forward(self, x, text_embeddings, yolo_embeddings):
            resnet_embeddings = self.up_until_embeddings(x)
            resnet_embeddings = resnet_embeddings.squeeze()
            final_embedding = torch.cat([resnet_embeddings, text_embeddings, yolo_embeddings], dim=1)
            outputs = self.classifier(final_embedding)
            return final_embedding, outputs

    model = ModifiedResnetwithEmbeddings(resnet=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT), number_features_embedding=768)
    model.load_state_dict(torch.load('resnet50_embeddings.pth', weights_only= True))
    model = model.to('cuda')

    train_inputs = []
    val_inputs = []
    for data, text_embedding, pose_embedding, label in train_dataloader:
        model.eval()
        data = data.to("cuda")
        label = label.to("cuda")
        text_embedding = text_embedding.to("cuda")
        pose_embedding = pose_embedding.to("cuda")
        SVM_input = model(data, text_embedding, pose_embedding)[0]
        label = label.view(-1, 1)  # reshape to [batch, 1] - column tensor
        SVM_input = torch.cat([SVM_input, label], dim=1)  # [64, features+1]
        SVM_input = SVM_input.to('cpu')
        SVM_input = SVM_input.detach().numpy()
        train_inputs.append(SVM_input)
    train_inputs = np.vstack(train_inputs)
    train_df = pd.DataFrame(train_inputs)

    for data, text_embedding,pose_embedding,label in val_dataloader:
        data = data.to("cuda")
        label = label.to("cuda")
        text_embedding = text_embedding.to("cuda")
        pose_embedding = pose_embedding.to("cuda")
        SVM_input = model(data, text_embedding, pose_embedding)[0]
        label = label.view(-1, 1)  # reshape to [batch, 1] - column tensor
        SVM_input = torch.cat([SVM_input, label], dim=1)  # [64, features+1]
        SVM_input = SVM_input.to('cpu')
        SVM_input = SVM_input.detach().numpy()
        val_inputs.append(SVM_input)

    val_inputs = np.vstack(val_inputs)
    val_df = pd.DataFrame(val_inputs)

    print(train_df)
    print("")
    print("")
    print("")
    print(val_df)

    # separarea datelor "de intrare" si "de iesire"
    X_train = train_df.iloc[:, :3428]
    Y_train = train_df[3428]
    X_test = val_df.iloc[:, :3428]
    Y_test = val_df[3428]

    param_grid = {
        'C': [2 ** i for i in range(-5, 8, 2)],
        'gamma': [2 ** i for i in range(-15, 4, 2)],
        'kernel': ['rbf']
    }

    # Grid search with cross-validation
    grid = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid.fit(X_train, Y_train)

    # Best parameters
    print(f"Best accuracy for SVM: {grid.best_score_}")
    print(f"Best parameters for SVM: {grid.best_params_}")

    y_pred = grid.best_estimator_.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Precision, Recall, F1 per class + macro/micro averages
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred, target_names=["NonFight", "Fight"]))

   # Accuracy
    print("Accuracy:", accuracy_score(Y_test, y_pred))
