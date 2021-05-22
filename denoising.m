foreground_color = double([0 0 255]);
background_color = double([245 210 110]);
centers = [foreground_color; background_color];


% Read image
img_path = 'denoise_input.jpg';
img = imread(img_path);
[height, width, channels] = size(img);

num_pixels = height*width;
num_neighbours = 2 * height * width - height - width;
Graph = BK_Create(num_pixels);


% Label 1 = Background, Label 2 = Foreground 
% holds cost to background/foreground
% First row == background cost
% Second row == foreground cost
pixels = double(reshape(img,[],3));
dist_background = sqrt(sum((pixels-foreground_color).^2,2));
dist_foreground = sqrt(sum((pixels-background_color).^2,2));
fb_cost = [dist_background, dist_foreground]';

% Compute neigh cost
m_lambda = 600;
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
pixels_r_diff = m_lambda .* mean(exp(-(sqrt((pixels(pixels_r_1(:),:) - pixels(pixels_r_2(:),:)).^2)./(2*img_var))),2);

% get pixels + bottom neighbour
pixels_b_1 = reshape((1:num_pixels-width),[],1);
pixels_b_2 = pixels_b_1 + height;
pixels_b_diff = m_lambda .* mean(exp(-(sqrt((pixels(pixels_b_1(:),:) - pixels(pixels_b_2(:),:)).^2)./(2*img_var))),2);

neigh_cost = [[pixels_r_1; pixels_b_1], [pixels_r_2; pixels_b_2], [pixels_r_diff; pixels_b_diff]];

% Set costs
neigh_cost = spconvert(neigh_cost);
neigh_cost(num_pixels,1)=0;

BK_SetNeighbors(Graph, neigh_cost);
BK_SetUnary(Graph, fb_cost);

% Get labelling
Energy = BK_Minimize(Graph)
Labeling = BK_GetLabeling(Graph);

labelled_img = reshape(Labeling,[height width]);
centers = reshape(centers, 2, []);
cmap = colormap(centers./255);
labelled_img = label2rgb(labelled_img, cmap);

% Show results
imshow(img)
figure;
imshow(labelled_img)
save_path = ['cleaned' num2str(m_lambda) '.jpg'];
imwrite(labelled_img,save_path)


