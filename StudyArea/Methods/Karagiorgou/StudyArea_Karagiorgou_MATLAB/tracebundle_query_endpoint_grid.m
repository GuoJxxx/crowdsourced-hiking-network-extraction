function candidate_indices = tracebundle_query_endpoint_grid(index, polygon_xy)
%TRACEBUNDLE_QUERY_ENDPOINT_GRID Return a sorted conservative candidate set.

candidate_indices = zeros(0, 1);
required_fields = {'format_version', 'cell_size_m', 'origin_x', 'origin_y', ...
    'grid_width', 'max_ix', 'max_iy', 'occupied_keys', 'offsets', ...
    'segment_ids'};
if ~isstruct(index) || ~all(isfield(index, required_fields)) || ...
        index.format_version ~= 1
    error('tracebundle:grid:InvalidIndex', 'Unsupported endpoint-grid index');
end
if ~isnumeric(polygon_xy) || ~isreal(polygon_xy) || ...
        ndims(polygon_xy) ~= 2 || size(polygon_xy, 1) ~= 2 || ...
        size(polygon_xy, 2) < 1 || any(~isfinite(polygon_xy(:)))
    error('tracebundle:grid:InvalidPolygon', ...
        'polygon_xy must be a finite real 2-by-M matrix');
end
if isempty(index.occupied_keys)
    return;
end

ix_min = floor((min(polygon_xy(1, :)) - index.origin_x) / index.cell_size_m);
ix_max = floor((max(polygon_xy(1, :)) - index.origin_x) / index.cell_size_m);
iy_min = floor((min(polygon_xy(2, :)) - index.origin_y) / index.cell_size_m);
iy_max = floor((max(polygon_xy(2, :)) - index.origin_y) / index.cell_size_m);

if ix_max < 0 || iy_max < 0 || ix_min > index.max_ix || iy_min > index.max_iy
    return;
end
ix_min = max(0, ix_min);
iy_min = max(0, iy_min);
ix_max = min(index.max_ix, ix_max);
iy_max = min(index.max_iy, iy_max);

[query_x, query_y] = meshgrid(ix_min:ix_max, iy_min:iy_max);
query_keys = uint64(query_x(:) + query_y(:) * index.grid_width + 1);
[found, locations] = ismember(query_keys, index.occupied_keys);
locations = locations(found);
if isempty(locations)
    return;
end

segments = zeros(0, 1);
for item = 1:numel(locations)
    first = index.offsets(locations(item));
    last = index.offsets(locations(item) + 1) - 1;
    segments = [segments; index.segment_ids(first:last)]; %#ok<AGROW>
end
candidate_indices = unique(segments, 'sorted');
candidate_indices = candidate_indices(:);
end
