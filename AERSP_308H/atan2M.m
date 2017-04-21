function [ T ] = atan2M(Y,X)
%Modified ATan2 maps from 0 to 2pi
    T=atan2(Y,X);
    [m,n]=size(T);
    for ii=1:m
        for jj=1:n
            if(T(ii,jj)<0) 
                T(ii,jj)=T(ii,jj)+2*pi;
            end
        end
    end

end

