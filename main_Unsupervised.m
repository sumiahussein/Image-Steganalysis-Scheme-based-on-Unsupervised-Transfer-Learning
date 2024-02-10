clc
clear all


tic
rng(2)
load('X.mat')
load('LB.mat')

X=double(X)/256;

Y=LB;

xs=[size(X,1),size(X,2)];
N=size(X,4);

YX=zeros(N,xs(1)*xs(2));

for i=1:N
    YX(i,:)=reshape(X(:,:,:,i),[1,128*128]);
end


layers1 = [ ...
    imageInputLayer([128 128])
    convolution2dLayer(126,126)
    leakyReluLayer
    maxPooling2dLayer(2)

    fullyConnectedLayer(128*128)
    regressionLayer];

layers2 = [ ...
    imageInputLayer([128 128])
    convolution2dLayer(126,126)
    leakyReluLayer
    maxPooling2dLayer(2)

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',100,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress' , ...
    'ExecutionEnvironment','cpu');


LB=categorical(Y);
c=0;
for i=1:1000:5000
    tr=i:(i+1000-1);
    c=c+1;
    net1{c} = trainNetwork(X(:,:,:,tr),YX(tr,:),layers1,options);
    tr=10000+(i:(i+1000-1));
    c=c+1;
    net1{c} = trainNetwork(X(:,:,:,tr),YX(tr,:),layers1,options);
end
save('net1','net1')


A=zeros(xs(1),xs(2),1,10000);
c=0;
for i=1:1000:5000
    tr=i:(i+1000-1);
    c=c+1;
    u=predict(net1{c},X(:,:,:,tr));
    A(:,:,1,tr)=reshape(u,[xs(1),xs(2)]);
     
    tr=10000+(i:(i+1000-1));
    c=c+1;
    u=predict(net1{c},X(:,:,:,tr));
    A(:,:,1,tr)=reshape(u,[xs(1),xs(2)]);
end

% train transfer learning CNN
tr=[1:5000,10001:15000];
ts=[5001:10000,15001:20000];
netX = trainNetwork(A(:,:,:,tr),LB(tr),layers2,options);

save('netX','netX')

TP=0;
TN=0;
FP=0;
FN=0;
Br=predict(netX,A(:,:,:,tr));
for i=1:size(tr,2)
    
    [B,I1]=max(Br(i,:));
    

    if(double(LB(tr(i)))==2)
        if(I1==2)
            TP=TP+1;
        else
            FN=FN+1;
        end
    else
        if(I1==1)
            TN=TN+1;
        else
            FP=FP+1;
        end
        
    end    
end

[TN TP FN FP]
Accuracy=(TP1+TN1)/(TP1+TN1+FN1+FP1);
Precision=TP1/(TP1+FP1);
DSC=(2*TP1)/(2*TP1+FN1+FP1);

fprintf('K-Fold 1: train Accuracy = %d\n',Accuracy);
fprintf('K-Fold 1: train Precision = %d\n',Precision);
fprintf('K-Fold 1: train DSC = %d\n',DSC);



As=zeros(xs(1),xs(2),1,10000);
c=0;
for i=5001:10000
    ts=i:(i+1000-1);
    c=c+1;
    u=predict(net1{c},X(:,:,:,tr));
    As(:,:,1,tr)=reshape(u,[xs(1),xs(2)]);
     
    tr=10000+(i:(i+1000-1));
    c=c+1;
    u=predict(net1{c},X(:,:,:,tr));
    As(:,:,1,tr)=reshape(u,[xs(1),xs(2)]);
end


TP=0;
TN=0;
FP=0;
FN=0;
Br=predict(net1,X(:,:,:,ts));
for i=1:size(tr,2)
    
    [B,I1]=max(Br(i,:));
    

    if(double(LB(tr(i)))==2)
        if(I1==2)
            TP=TP+1;
        else
            FN=FN+1;
        end
    else
        if(I1==1)
            TN=TN+1;
        else
            FP=FP+1;
        end
        
    end    
end
