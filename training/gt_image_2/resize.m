rgb = imread('img1.jpg');
rotated = imrotate(rgb,-90);
gray = rgb2gray(rotated);
se = strel('disk',12);
tophatFiltered = imtophat(gray,se);
contrastAdjusted = imadjust(tophatFiltered);
thresh = graythresh(contrastAdjusted);
BW = imbinarize(contrastAdjusted,thresh);
imshow(BW)