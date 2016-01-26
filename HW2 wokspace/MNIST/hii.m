%for 2 h ii
clc
close all
clear
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
numofz=30;
%30
numofinput=785;
alpha3=1e-5;
%1e-5
alpha2=8e-4;
%8e-4
alpha1=1e-4;
%1e-4

rand('seed',0);
w1=rand(numofinput,numofhidden-1);%w1 785 * 784
rand('seed',0);
w2=rand(numofhidden,numofz-1);%w2 785 * 10
rand('seed',0);
w3=rand(numofz,numofoutput);

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

h = repmat([1 zeros(1,numofhidden-1)],numoftrain,1);
z = repmat([1 zeros(1,numofz-1)],numoftrain,1);
y = zeros(numoftrain,numofoutput);
acc=[];
acc_h=[];
acc_t=[];
for j=1:500
    %forward activation.
    h(:,2:numofhidden)=sigmf(train * w1,[1 0]);
    z(:,2:numofz)=sigmf(h * w2,[1 0]);
    a=exp(z * w3);
    dev=sum(a,2)*ones(1,10);
    y= a./dev;
    [~, index] = max(y, [], 2);
    count=0;
    for i=1:length(y)
        if(index(i)==train_labels(i)+1)
            count=count+1;
        end
    end
    pred=count/length(y)
    acc=[acc pred];
    acc_h=[acc_h preddouble(numofhidden,numofz,holdon,w1,w2,w3,hold_labels)];
    acc_t=[acc_t preddouble(numofhidden,numofz,test,w1,w2,w3,test_labels)];
    
    %back propagation.
    backward1=trl-y;
    w3=w3+alpha3.* (z'*backward1);
    backward2=sigmf(h*w2,[1 0]).*sigmf(-h*w2,[1 0]).*(backward1*(w3(2:numofz,:))');
    w2=w2+alpha2.*(h' * backward2);
    backward3=sigmf(train*w1,[1 0]).*sigmf(-train*w1,[1 0]).*(backward2*(w2(2:numofhidden,:))');
    w1=w1+ alpha1.* (train' *backward3);
end

hold all
plot(acc);
plot(acc_h);
plot(acc_t);