%for 2f
clc
close all
clear


%mu=0.00000001;
mu=0.9;
numoftrain=50000;
train = loadMNISTImages('train-images-idx3-ubyte');
holdon=[ones(10000,1) zscore(train(:,1:10000))'];
train = [ones(numoftrain,1) zscore(train(:, 10001:10000+numoftrain))'];% every row is a picture. 60000*785
test = loadMNISTImages('t10k-images-idx3-ubyte');
test = [ones(2000,1) zscore(test(:, 1:2000))'];% every row is a picture 2000*784
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
hold_labels = train_labels(1:10000);
train_labels = train_labels(10001:10000+numoftrain);
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
test_labels = test_labels(1:2000);

numofoutput=10;
numofhidden=30;
%30
numofinput=785;
alpha=0.00003;
%0.00002

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
prev22=zeros(size(w2));
prev21=zeros(size(w2));
prev11=zeros(size(w1));
prev12=zeros(size(w1));
for j=1:2000
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
    acc_h=[acc_h predict(numofhidden,holdon,w1,w2,hold_labels,'sigmoid')];
    acc_t=[acc_t predict(numofhidden,test,w1,w2,test_labels,'sigmoid')];
    pred=count/length(y)
    acc=[acc pred];
    %back propagation.
    prev21=alpha.* (z'*(trl-y));
    w2=w2+prev21+mu*prev22;
    prev22=prev21;
    %w2 785*10 z' 785*60000  y 60000*10
    backward=sigmf(train*w1,[1 0]).*sigmf(-train*w1,[1 0]).*((trl-y)*(w2(2:numofhidden,:))');
    %train 60000*785 w1 785*784 
    prev11=alpha.* (train' *backward);
    w1=w1+ prev11+mu*prev12;
    prev12=prev11;
end

hold all
plot(acc,'b');
plot(acc_h,'r');
plot(acc_t,'g');
[ma,index]=max(acc_h)
acc_t(index)
