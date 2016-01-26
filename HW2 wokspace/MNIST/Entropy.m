function E = Entropy(numofhidden,w1, w2, train, trl)
    z(:,2:numofhidden)=sigmf(train * w1,[1 0]);
    z(:,1)=ones(length(trl),1);
    a=exp(z * w2);
    dev=sum(a,2)*ones(1,10);
    y= a./dev;
    E = -sum(sum(trl .* log(y)));
end