import torch
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
from ultralytics import YOLO

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

class EarlyStopping:         # antrenarea se opreste cand apar 2 epoci in care nu mai creste val_loss
    def __init__(self, patience=20, delta=0.001, checkpoint_path="resnet50_embeddings.pth"):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model to disk
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == '__main__':
    train_dataset = PicturesDatasetEmbeddings(r'D:\Downloads\database\RWF-2000\train', 'embeddings_from_gemini_768_no_keywords.csv', transform = transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
    val_dataset = PicturesDatasetEmbeddings(r'D:\Downloads\database\RWF-2000\val', 'embeddings_from_gemini_768val_no_keywords.csv', transform = transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


    class ModifiedResnetwithEmbeddings(torch.nn.Module):
        def __init__(self, resnet, number_features_embedding):
            super().__init__()
            self.up_until_embeddings = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features = resnet.fc.in_features + 768 + 612, out_features = 1024),  # expand or compress, your call
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


    #train loop
    resnet_with_embeddings = ModifiedResnetwithEmbeddings(torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT), 768).to("cuda")
    resnet_with_embeddings.load_state_dict(torch.load('resnet50_embeddings.pth', weights_only = True))
    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer =  torch.optim.SGD(resnet_with_embeddings.parameters(),lr=0.0001, momentum = 0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,  # halve LR
    #     patience=1  # wait 1 epoch with no improvement
    # )
    # scheduler.get_last_lr()
    # last_epoch = 0
    # train_losses = []
    # val_losses = []
    # nr_epoci = 80           #maxim
    # early_stopping = EarlyStopping()
    # for epoch in range(nr_epoci):
    #     train_loss = 0
    #     val_loss = 0
    #     for data, text_embedding, pose_embedding, label in train_dataloader:
    #         data = data.to("cuda")
    #         label = label.to("cuda")
    #         text_embedding = text_embedding.to("cuda")
    #         pose_embedding = pose_embedding.to("cuda")
    #         optimizer.zero_grad()
    #         resnet_with_embeddings.train()
    #         output = resnet_with_embeddings(data, text_embedding, pose_embedding)[1]
    #         loss = loss_function(output, label)
    #         train_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #     train_loss /= len(train_dataloader)
    #     train_losses.append(train_loss)
    #     for data, text_embedding, pose_embedding, label in val_dataloader:
    #         with torch.no_grad():
    #             resnet_with_embeddings.eval()
    #             data = data.to("cuda")
    #             label = label.to("cuda")
    #             text_embedding = text_embedding.to("cuda")
    #             pose_embedding = pose_embedding.to("cuda")
    #             output = resnet_with_embeddings(data, text_embedding, pose_embedding)[1]
    #             loss = loss_function(output, label)
    #             val_loss += loss.item()
    #     val_loss /= len(val_dataloader)
    #     scheduler.step(val_loss)
    #     val_losses.append(val_loss)
    #     print(f"Epoca {epoch}:")
    #     print(f"Train loss: {train_loss:.4f}")
    #     print(f"Val_loss: {val_loss:.4f}")
    #     early_stopping(val_loss, resnet_with_embeddings)
    #     last_epoch = epoch
    #     if early_stopping.early_stop:
    #         print(f"Stopped early at epoch {epoch} - loss hasn't improved for {early_stopping.patience} epochs")
    #         print("")
    #         break

    # computing accuracy

    correct = 0
    total = 0
    resnet_with_embeddings.eval()
    with torch.no_grad():
        for data, text_embedding, pose_embedding, label in val_dataloader:
            data = data.to("cuda")
            label = label.to("cuda")
            text_embedding = text_embedding.to("cuda")
            pose_embedding = pose_embedding.to("cuda")
            output = resnet_with_embeddings(data, text_embedding, pose_embedding)[1]
            predicted = torch.argmax(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy}")
    print("")

    # computing confusion matrix
    true_negative = 0  # TN - true negative
    true_positive = 0  # TP - true positive
    false_positive = 0  # FP - false positive
    false_negative = 0  # FN - false negative
    with torch.no_grad():
        for data, text_embedding, pose_embedding, label in val_dataloader:
            data = data.to("cuda")
            label = label.to("cuda")
            text_embedding = text_embedding.to("cuda")
            pose_embedding = pose_embedding.to("cuda")
            output = resnet_with_embeddings(data, text_embedding, pose_embedding)[1]
            predicted = torch.argmax(output, 1)
            true_negative += ((label == 0) & (predicted == 0)).sum().item()
            true_positive += ((label == 1) & (predicted == 1)).sum().item()
            false_negative += ((label == 1) & (predicted == 0)).sum().item()
            false_positive += ((label == 0) & (predicted == 1)).sum().item()

    precision0 = true_negative / (true_negative + false_negative)
    precision1 = true_positive / (true_positive + false_positive)
    recall0 = true_negative / (true_negative + false_positive)
    recall1 = true_positive / (true_positive + false_negative)

    print("confusion_matrix")
    print(true_negative, false_positive)
    print(false_negative, true_positive)
    print('')
    print(f"Precision for class 0 = {precision0}")
    print(f"Recall for class 0 = {recall0}")
    print(f"Precision for class 1 = {precision1}")
    print(f"Recall for class 1 = {recall1}")

    # plt.plot([i for i in range (1, last_epoch+2)],train_losses)
    # plt.plot([i for i in range (1, last_epoch+2)], val_losses)
    # plt.show()


   #torch.save(resnet_with_embeddings.state_dict(), "resnet_with_embeddings2.pth")




