function ipts=tb_pair_intersections(mpts,aa,p1)
% Batch the public pair-intersection calls, preserving b/c and append order.
% Use the same matGeom functions, angle thresholds and distance predicate.
n=size(mpts,1);
ipts=aa(1,1:2);
lines=createLine(mpts(:,1:2),mpts(:,3:4));
degrees=rad2deg(angle2Points(mpts(:,1:2),mpts(:,3:4)));
for b=1:2:n-1
    ipts=[ipts; mpts(b,3:4)]; %#ok<AGROW>
    candidates=(b+2):2:(n-2);
    for offset=1:4096:numel(candidates)
        c=candidates(offset:min(offset+4095,numel(candidates)));
        if isempty(c), continue; end
        c=c(:);
        d13=abs(degrees(b)-degrees(c));
        d14=abs(degrees(b)-degrees(c+1));
        d23=abs(degrees(b+1)-degrees(c));
        d24=abs(degrees(b+1)-degrees(c+1));
        valid=[d13>=25 & d13<=320, d14>=25 & d14<=320, ...
               d23>=25 & d23<=320, d24>=45 & d24<=320];
        points=cat(3,intersectLines(lines(b,:),lines(c,:)), ...
                     intersectLines(lines(b,:),lines(c+1,:)), ...
                     intersectLines(lines(b+1,:),lines(c,:)), ...
                     intersectLines(lines(b+1,:),lines(c+1,:)));
        % c1:type1,type2,type3,type4; c2:type1,... matches the public appends.
        points=reshape(permute(points,[3 1 2]),[],2);
        valid=reshape(valid.',[],1);
        valid=valid & ~isnan(points(:,1)) & ~isinf(points(:,1));
        ds=distancePoints(aa(1,1:2),points,2);
        valid=valid & ds(:)<=p1;
        ipts=[ipts;points(valid,:)]; %#ok<AGROW>
    end
end
end
