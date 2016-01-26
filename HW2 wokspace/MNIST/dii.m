%for 2d ii
clc
close all
clear
numoftrain=500;
train = loadMNISTImages('train-images-idx3-ubyte');
train = [ones(numoftrain,1) zscore(train(:, 1:numoftrain))'];% every row is a picture. 60000*785
test = loadMNISTImages('t10k-images-idx3-ubyte');
test = [ones(2000,1) zscore(test(:, 1:2000))'];% every row is a picture 2000*784
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
train_labels = train_labels(1:numoftrain);
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
test_labels = test_labels(1:2000);

numofoutput=10;
numofhidden=30;
numofinput=785;
alpha=0.00002;
epsilon=1e-5;

rand('seed',1);
w1=rand(numofinput,numofhidden-1);%w1 785 * 784
rand('seed',1);
w2=rand(numofhidden,numofoutput);%w2 785 * 10

tel=zeros(length(test_labels),10);
trl=zeros(length(train_labels),10);

for i=0:9
    for j=1:length(train_labels)
        trl(j,i+1)=(train_labels(j)==i);
    end
    for j=1:length(test_labels)
        tel(j,i+1)=(test_labels(j)==i);
    end
end
%trl 60000*10
%tel 2000*10

z = repmat([1 zeros(1,numofhidden-1)],numoftrain,1);
y = zeros(numoftrain,numofoutput);
acc=[];
acc_h=[];
acc_t=[];
col1=[];
col2=[];
for j=1:3
    %forward activation.
    z(:,2:numofhidden)=sigmf(train * w1,[1 0]);
    %train 60000*785 w1 785input * 784hidden
    a=exp(z * w2);
    % z 60000*785 w2 785*10
    dev=sum(a,2)*ones(1,10);
    y= a./dev;
    %y 60000 * 10
    [~, index] = max(y, [], 2);
    count=0;
    for i=1:length(y)
        if(index(i)==train_labels(i)+1)
            count=count+1;
        end
    end
    pred=count/length(y);
    acc=[acc pred];
    %back propagation.
    %E = Entropy(numofhidden,w1, w2, train, trl);
    E1=zeros(numofinput,numofhidden-1);
    E2=zeros(numofhidden,numofoutput);
    for i=1:numofinput
        for j=1:numofhidden-1
            ww1=w1;
            ww2=w1;
            ww1(i,j)=ww1(i,j)+epsilon;
            ww2(i,j)=ww2(i,j)-epsilon;
            E1(i,j)=(Entropy(numofhidden,ww1,w2,train,trl)-Entropy(numofhidden,ww2,w2,train,trl))/(2*epsilon);
        end
    end
    for i=1:numofhidden
        for j=1:numofoutput
            ww1=w2;
            ww2=w2;
            ww1(i,j)=ww1(i,j)+epsilon;
            ww2(i,j)=ww2(i,j)-epsilon;
            E2(i,j)=(Entropy(numofhidden,w1,ww1,train,trl)-Entropy(numofhidden,w1,ww2,train,trl))/(2*epsilon);
        end
    end    
    grad1=-z'*(trl-y);
    w2=w2+alpha.* (z'*(trl-y));
    backward=sigmf(train*w1,[1 0]).*sigmf(-train*w1,[1 0]).*((trl-y)*(w2(2:numofhidden,:))');
    w1=w1+ alpha.* (train' *backward);
    grad2=-train'*backward;
    
    col1=[col1 sum(sum(abs(E2-grad1)))];
    col2=[col2 sum(sum(abs(E1-grad2)))];
end

