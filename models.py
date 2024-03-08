import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, weights = None):
        super(SiameseNetwork, self).__init__()
        
        if(weights!=None):
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_classes = 2
            num_features = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_features, num_classes)
            self.resnet50.load_state_dict(torch.load(weights))
        else:
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(2052, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward_once(self, view_data):
        x, boxes, _ = view_data
        # import pdb; pdb.set_trace()
        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        x = torch.cat((x, boxes[:,:4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        distance = F.pairwise_distance(output1, output2)
        norm_distance = 1.0/ (1.0 + distance)
        return distance, norm_distance
    


class Pseudo_attn(nn.Module):
    def __init__(self, weights = None):
        super(Pseudo_attn, self).__init__()
        
        if(weights!=None):
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_classes = 2
            num_features = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_features, num_classes)
            self.resnet50.load_state_dict(torch.load(weights))
        else:
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(2052, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)


    def forward_once(self, view_data):
        x, boxes, _ = view_data
        # import pdb; pdb.set_trace()
        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        x = torch.cat((x, boxes[:,:4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.normalize(x, p=2, dim=1)


    def forward(self, view_0, view_1):
        # import pdb; pdb.set_trace()
        num_view0_props = view_0[0].shape[0]
        
        batched_view1_imgs = view_1[0].repeat(num_view0_props,1,1,1)
        batched_view1_prop = view_1[1].repeat(num_view0_props,1)
        input1 = (batched_view1_imgs, batched_view1_prop, view_1[2])
        
        batched_view0_imgs = view_0[0].repeat_interleave(num_view0_props,dim=0)
        batched_view0_prop = view_0[1].repeat_interleave(num_view0_props,dim=0)
        input0 = (batched_view0_imgs, batched_view0_prop, view_0[2])
                
        output1 = self.forward_once(input0)
        output2 = self.forward_once(input1)
        context = F.cosine_similarity(output1, output2, dim=1)
        context = context * batched_view1_prop[:,4]
        context = sum(torch.chunk(context, chunks=num_view0_props))

        # max_values = []
        # chunks = torch.chunk(context, chunks=num_view0_props)
        # for chunk in chunks:
        #     chunk_max, _ = torch.max(chunk, dim=0)
        #     max_values.append(chunk_max)
        # context = torch.stack(max_values, dim=0)

        preds = view_0[1][:,4] + context
        return preds

class MAX_model(nn.Module):
    def __init__(self, weights = None):
        super(MAX_model, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_classes = 2
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)        
        if(weights!=None):
            self.resnet50.load_state_dict(torch.load(weights))
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(2052, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)


    def forward_once(self, view_data):
        x, boxes, _ = view_data
        # import pdb; pdb.set_trace()
        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        x = torch.cat((x, boxes[:,:4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.normalize(x, p=2, dim=1)


    def forward(self, view_0, view_1):
        # import pdb; pdb.set_trace()
        num_view0_props = view_0[0].shape[0]
        
        
        embedd_0 = self.forward_once(view_0)
        embedd_1 = self.forward_once(view_1)
        
        assert embedd_0.size() == embedd_1.size()

        # Max model
        context = torch.matmul(embedd_0, embedd_1.transpose(-1, -2))
        preds, _ = torch.max(context * view_1[1][:,4].unsqueeze(1), axis=1) 
        preds = preds + view_0[1][:,4]

        return preds




class Siamese_Attn(nn.Module):
    def __init__(self, weights = None, embed_size=64, dropout = 0.2):
        super(Siamese_Attn, self).__init__()
        
        if(weights!=None):
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_classes = 2
            num_features = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_features, num_classes)
            self.resnet50.load_state_dict(torch.load(weights))
        else:
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        self.fc1 = nn.Linear(2052, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        
        self.Q = nn.Linear(256, embed_size)
        self.K = nn.Linear(256, embed_size)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.scale = embed_size ** -0.5
        

    def forward_once(self, view_data):
        x, boxes, _ = view_data
        # import pdb; pdb.set_trace()
        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        x = torch.cat((x, boxes[:,:4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.relu(self.fc3(x))
        return x
    

    def forward(self, view_0, view_1):
        # import pdb; pdb.set_trace()
        input_0 = self.forward_once(view_0)
        input_1 = self.forward_once(view_1)
        
        query = self.Q(input_0)
        keys = self.K(input_1)
        
        dots = torch.matmul(query, keys.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        context = torch.matmul(attn, view_1[1][:,4].unsqueeze(1))

        preds = view_0[1][:,4] + context.squeeze()
        
        return preds
    
