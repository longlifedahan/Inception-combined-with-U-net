class Inception_base(nn.Module):
    '''
    param-depth_dim:    channel dim
    param-input_size:   input
    param-config:       params
    '''
    # [B,4,T,257] <=> [B C H W]
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()
        self.depth_dim = depth_dim      
        # 【1*1 conv】
        #  [B,4,T,257]->[B,2,T,257]
        self.conv1_1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.BN1_1 = nn.BatchNorm2d(config[0][0],track_running_stats=False)
        self.Relu1 = nn.PReLU(config[0][0])         
        # 【3*3 conv】
        # [B,4,T,257]->[B, 4, T, 257]->[B, 8, T, 257]
        self.conv2_1 = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.BN2_1 = nn.BatchNorm2d(config[1][0],track_running_stats=False)
        self.Relu2_1 = nn.PReLU(config[1][0])
        self.conv2_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=(3, 3), stride=(1, 1), padding=1)  
        self.Relu2_3 = nn.PReLU(config[1][1])        
        # 【5*5 conv】
        # [B,4,T,257]->[B, 2, T, 257]->[B, 4, T, 257]
        self.conv3_1 = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.BN3_1 = nn.BatchNorm2d(config[2][0],track_running_stats=False)
        self.Relu3_1 = nn.PReLU(config[2][0])
        self.conv3_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=(5, 5), stride=(1, 1), padding=2) 
        self.Relu3_5 = nn.PReLU(config[2][1])        
        # 【maxpool】
        # [B,4,T,257]->[B, 4, T, 257]->[B, 2, T, 257]
        self.max_pool_4_m = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1) 
        self.BN4_m = nn.BatchNorm2d(input_size,track_running_stats=False) 
        self.Relu4_m = nn.PReLU(input_size)
        self.conv_4_1 = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.Relu4_1 = nn.PReLU(config[3][1])
    def forward(self,input):
        output1=self.Relu1(self.BN1_1(self.conv1_1(input)))         # 1*1 con+BN
        output2=self.Relu2_1(self.BN2_1(self.conv2_1(input)))       # 1*1 conv+BN+3*3 conv
        output2=self.Relu2_3(self.conv2_3(output2))
        output3=self.Relu3_1(self.BN3_1(self.conv3_1(input)))       # 1*1 conv+BN+5*5 conv
        output3=self.Relu3_5(self.conv3_5(output3))
        output4=self.Relu4_m(self.BN4_m(self.max_pool_4_m(input)))  # max-pool+1*1conv
        output4=self.Relu4_1(self.conv_4_1(output4))
        output=torch.cat([output1,output2,output3,output4],dim=self.depth_dim) # concant at dim 1
        return output

class Inception_Unet(nn.Module):
    '''
    Inception combined with U_net
    basic/stft     [b,4,f,257]
    humanears/GFCC [b,4,f,37]
    '''
    def __init__(self):
        super(Inception_Unet,self).__init__()
        print("#### Model Initialize:Inception_Unet[track_running_stats=False,combined with U-net] ####")   

        # process basic spectrogram with Inception
        self.inception1=Inception_base(depth_dim=1,input_size=4,config=[[1],[2,4],[1,2],[3,1]])
        self.inception2=Inception_base(depth_dim=1,input_size=8,config=[[2],[4,8],[2,4],[3,2]])
        self.inception3=Inception_base(depth_dim=1,input_size=16,config=[[4],[8,16],[4,8],[3,4]])
        self.inception4=Inception_base(depth_dim=1,input_size=32,config=[[4],[8,16],[4,8],[3,4]])
        self.inception5=Inception_base(depth_dim=1,input_size=32,config=[[8],[16,32],[8,16],[3,8]])
        self.inception6=Inception_base(depth_dim=1,input_size=64,config=[[8],[16,32],[8,16],[3,8]]) 

        # process GFCC features and do down samples
        self.conv1=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu1=nn.PReLU(8)
        self.max_pool1=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2),padding=0)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu2=nn.PReLU(16)
        self.max_pool2=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2),padding=0)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu3=nn.PReLU(32)
        self.max_pool3=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2),padding=0)
        self.conv4=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu4=nn.PReLU(32)
        self.max_pool4=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2),padding=0)
        self.conv5=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu5=nn.PReLU(64)
        self.max_pool5=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2),padding=0)
        self.conv6=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1),padding=0)
        self.relu6=nn.PReLU(64)
        self.max_pool6=nn.MaxPool2d(kernel_size=(1,1),stride=(1,1),padding=0)
        
        # process outputs, SE
        self.max_pool=nn.MaxPool2d(kernel_size=(1,4),stride=(1,4),padding=0)    # [B,64,T,k]->[B,64,T,k//4]
        self.conv=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1))    # 1*1 conv [B,64,T,k//4]->[B,1,T,k//4]
        self.BN=nn.BatchNorm2d(1,track_running_stats=False)                     # BN
        self.Relu=nn.PReLU(1)                                                   # 1 channels left

    def forward(self,input):
        ori=input[:,:,:,:257]
        human=input[:,:,:,257:]
        # L1 [b,4,t,257]+[b,4,t,37]->[b,8,t,18]=[b,4,t,275]
        ori=self.inception1(ori)
        human=self.max_pool1(self.relu1(self.conv1(human)))
        ori=torch.cat([ori,human],dim=-1)
        # L2 [b,4,t,275]+[b,8,t,18]->[b,8,t,9]=[b,8,t,284]
        ori=self.inception2(ori)
        human=self.max_pool2(self.relu2(self.conv2(human)))
        ori=torch.cat([ori,human],dim=-1)
        # L3 [b,8,t,284]+[b,16,t,9]->[b,8,t,4]=[b,16,t,288]
        ori=self.inception3(ori)
        human=self.max_pool3(self.relu3(self.conv3(human)))
        ori=torch.cat([ori,human],dim=-1)
        # L4 [b,16,t,288]+[b,32,t,4]->[b,16,t,2]=[b,32,t,290]
        ori=self.inception4(ori)
        human=self.max_pool4(self.relu4(self.conv4(human)))
        ori=torch.cat([ori,human],dim=-1)
        # L5 [b,32,t,290]+[b,32,t,2]->[b,32,t,1]=[b,64,t,291]
        ori=self.inception5(ori)
        human=self.max_pool5(self.relu5(self.conv5(human)))
        ori=torch.cat([ori,human],dim=-1)
        # L6 [b,64,t,291]+[b,64,t,1]->[b,64,t,1]=[b,64,t,292]
        ori=self.inception6(ori)
        human=self.max_pool6(self.relu6(self.conv6(human)))
        ori=torch.cat([ori,human],dim=-1)
        # output [b,64,t,292]->[b,64,t,73]->[b,t,73]
        output=self.max_pool(ori)
        output=self.Relu(self.BN(self.conv(output)))
        output=torch.squeeze(output,dim=1)
        return output
