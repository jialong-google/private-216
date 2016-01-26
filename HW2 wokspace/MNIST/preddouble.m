function [ pred ] = preddouble( numofhidden,numofz,train,w1,w2,w3,train_labels )
    h = repmat([1 zeros(1,numofhidden-1)],length(train_labels),1);
    z = repmat([1 zeros(1,numofz-1)],length(train_labels),1);
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
    pred=count/length(y);

end

