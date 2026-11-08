function [inside,boundary]=tb_cached_inpolygon(x,y,px,py)
% Cache only exact coordinate arrays. Any mutation triggers index rebuilding.
% Geometric membership remains the public MATLAB inpolygon computation.
persistent entries
if numel(x)<256 || ~isreal(x) || ~isreal(y)
    [inside,boundary]=inpolygon(x,y,px,py); return;
end
if isempty(entries), entries={}; end
for k=1:numel(entries)
    entry=entries{k};
    if isequal(x,entry.x) && isequal(y,entry.y)
        [inside,boundary]=tb_indexed_inpolygon(entry.index,x,y,px,py); return;
    end
end
if any(~isfinite(x(:))) || any(~isfinite(y(:)))
    [inside,boundary]=inpolygon(x,y,px,py); return;
end
index=tracebundle_build_endpoint_grid([x(:) y(:) x(:) y(:)],200.0);
entry=struct('x',x,'y',y,'index',index);
entries=[{entry} entries];
if numel(entries)>4, entries=entries(1:4); end
[inside,boundary]=tb_indexed_inpolygon(index,x,y,px,py);
end
