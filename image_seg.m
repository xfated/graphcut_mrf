img_path = 'part2_images/12003';
img = imread([img_path '.jpg']);

num_clusters = 3;
num_iter = 1000;
num_init = num_clusters;

% Get centers of size NumClusters x Channels
[labels, centers] = k_means(img,num_clusters,num_iter,num_init);
'Centers obtained'
centers
addpath('gco-v3.0/matlab');
% Create graph
[height width channels] = size(img);
num_pixels = height * width;

pixels = double(reshape(img, [],3));
% Compute data cost [labels x num_pixels]
% distance between pixels (num_pixels x 3) to 
% cluster centers (num_clusters x 3)
pixels_data = reshape(pixels, num_pixels, 1, 3);
pixels_data = repmat(pixels_data, 1, num_clusters, 1); % num_pixels x num_clusters x 3
centers = reshape(centers, [], num_clusters, 3); % 1 x num_clusters x 3
pixel_dist = pixels_data - centers;
pixel_dist = pixel_dist.^2;
pixel_dist = sum(pixel_dist, 3);
data_cost = pixel_dist';

% Compute Smooth cost
smooth_cost = ones(num_clusters) - diag(ones([1 num_clusters]));

% Compute Neighbour cost
img_var = var(pixels,0,1);
% Get pixels, right + neighbour
num_pixels_r = height * (width - 1);
pixels_r_1_temp = ones(height,width);
count = 1;
for row = 1:height
    for column = 1:width
        pixels_r_1_temp(row, column) = count;
        count = count + 1;
    end
end
pixels_r_1 = reshape(pixels_r_1_temp(:,1:width-1),[],1);
pixels_r_2 = pixels_r_1 + 1;
pixels_r_diff = mean(exp(-(sqrt((pixels(pixels_r_1(:),:) - pixels(pixels_r_2(:),:)).^2)./(2*img_var))),2);

% get pixels + bottom neighbour
pixels_b_1 = reshape((1:num_pixels-width),[],1);
pixels_b_2 = pixels_b_1 + height;
pixels_b_diff = mean(exp(-(sqrt((pixels(pixels_b_1(:),:) - pixels(pixels_b_2(:),:)).^2)./(2*img_var))),2);

neigh_cost = [[pixels_r_1; pixels_b_1], [pixels_r_2; pixels_b_2], [pixels_r_diff; pixels_b_diff]];

neigh_cost = spconvert(neigh_cost);
neigh_cost(num_pixels,1) = 0;

% GCO Lib
Graph = GCO_Create(num_pixels,num_clusters);
GCO_SetDataCost(Graph,data_cost);
GCO_SetNeighbors(Graph,neigh_cost);
GCO_SetSmoothCost(Graph,smooth_cost);
GCO_SetLabelOrder(Graph,randperm(num_clusters))
GCO_Expansion(Graph);
final_label = GCO_GetLabeling(Graph);
GCO_Delete(Graph);

% Plot segmentation
final_label_img = reshape(final_label, height, width);
centers = reshape(centers, num_clusters,[]);
cmap = colormap(centers./255);
seg_img = label2rgb(final_label_img, cmap);
% imshow(img);
% figure()
imshow(seg_img);

% for comparison
kmean_label_img = reshape(labels,height,width);
kmean_img = label2rgb(kmean_label_img,cmap);
% figure()
% imshow(kmean_img);

save_path =[img_path '_' num2str(num_clusters) '.jpg'];
% imwrite(seg_img,save_path);
