function [ pred ] = predict(numofhidden,train,w1,w2,train_labels, prediction)
    z=zeros(length(train_labels),numofhidden);
    switch(prediction)
        case 'sigmoid'
            z(:,2:numofhidden)=sigmf(train * w1,[1 0]);
        case 'relu'
            z(:,2:numofhidden)=max(train * w1,zeros(length(train_labels),numofhidden-1));
        case 'tanh'
            z(:,2:numofhidden)=tanh(train * w1);
    end
    z(:,1)=ones(length(train_labels),1);
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
end

