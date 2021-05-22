function [labels, best_centers] = k_means(img, K, iters, num_tries)
% Author: Kai Yang
% @img: Image to do clustering on
% @K: Number of clusters
% @iters: Number of iterations to compute cluster center
% @num_tries: Number of times to repeat algo. (try different random init)
%
% Computes K-mean cluster centers
%

% Get image size
img = int32(img);
img_size = size(img);
num_pixels = img_size(1)*img_size(2);
channels = 3;
if length(img_size) < 3
    channels=1;
end

% Flatten to num_pixels x channels
pixels = reshape(img, num_pixels, channels);

best_centers = zeros([K 3]);
best_loss = 100000000000;
best_labels = zeros([num_pixels 1]); 
labels = zeros([num_pixels 1]);

for x = 1:num_tries
    for iter = 1:iters
        % Random init
        random_init = randi([1 num_pixels],K);
        centers = zeros([K 3]);
        for i = 1:K
            centers(i,:) = pixels(random_init(i),:);
        end
        
        % Get labels
        [loss, new_labels] = assign_centroids(pixels, centers, K);
        % Get labels
        centers = recomputeCentroid(pixels, labels, K);
       
        if sum(new_labels ~= labels) == 0
            %'Early exit'
            break
        end
        labels = new_labels;
        
    end
    if loss < best_loss && sum(isnan(centers),'all')==0
        %'Updating best centers'
        best_loss = loss;
        best_centers = centers;
        best_labels = labels;
    end
end

end
%% Helper functions
function [loss, labels] = assign_centroids(pixels, centers, K)
    % Size pixels: N x 3
    % Size centers: K x 3
    pixels_size = size(pixels);
    if length(pixels_size)==2
        % Size pixels: N x K x 3
        pixels = reshape(pixels, pixels_size(1),1,3);
        pixels = repmat(pixels,1,K,1);
        
        centers = reshape(centers, 1, K, 3);
    end
   
    % Calc pixel dist from cluster
    pixel_dist = double(pixels) - centers;
    pixel_dist = pixel_dist.^2;
    pixel_dist = sum(pixel_dist, 3);
    [M labels] = min(pixel_dist,[],2);
    
    loss = sum(M,'all');
end

function centers = recomputeCentroid(pixels, labels, K)
    % Size pixels: N x 3
    % Size labels: N x 1
    % Size K: scalar
    
    % Prepare one hot
    one_hot = int32(labels==1:K);
    % Check for num channels
    if length(size(pixels)) == 2
        one_hot = repmat(one_hot,1,1,3);
        
        % Prepare pixels
        pixels_size = size(pixels);
        pixels = reshape(pixels, pixels_size(1), 1, 3);
    end
    
    % pixels: num_pixels x 3, one_hot: num_pixels x K x 3
    pixels = pixels .* one_hot;
    
    % Count num in each label
    pixel_count = sum(one_hot, 1);

    % Sum all pixels with label
    centers = sum(pixels, 1);

    % Get average of cluster
    centers = centers ./ pixel_count;
    centers = reshape(centers, K, []);
end