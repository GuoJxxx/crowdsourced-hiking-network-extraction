function [inside,boundary]=tb_indexed_inpolygon(index,x,y,px,py)
% Conservative indexed candidates, then the unchanged public inpolygon test.
if any(~isfinite(px(:))) || any(~isfinite(py(:)))
    [inside,boundary]=inpolygon(x,y,px,py); return;
end
candidates=tracebundle_query_endpoint_grid(index,[px(:).';py(:).']);
inside=false(size(x)); boundary=false(size(x));
if isempty(candidates), return; end
[a,b]=inpolygon(x(candidates),y(candidates),px,py);
inside(candidates)=a; boundary(candidates)=b;
end
