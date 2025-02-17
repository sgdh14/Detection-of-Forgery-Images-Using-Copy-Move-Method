function IT_Project_2024()
clc
clear
close all
addpath('1_Test_Images');
%% Parameters
CASE=1    
s_threshold=0.05;KPlimit=1;nKP1=0; nKP2=0;

%% Part-1: Read Image Data
        [name,path]=uigetfile('*.*');
        CI=imread([path,name]);
        figure(1), imshow(CI);title('Colour Image');
        [height,width,numChannels] = size(CI);
        if numChannels > 1
            GI = rgb2gray(CI);
        else
            GI = CI;
        end
        figure(2),imshow(GI),title('Gray Scale Image');
%%  Identification of SURF KPs      
        patches=[];
        pts1=detectSURFFeatures(GI);
        nKP1=pts1.Count;
        strong1 = pts1;
        if nKP1>0
            KP1=abs([pts1.Location]);
            figure(3),imshow(GI),title('Image with SURF-KPs');hold on;
            gscatter(KP1(:,1),KP1(:,2),[],[],[],[],'off'); 
        end
        fprintf('\nNo of SURF KPs: %d  ',nKP1);
        
        %% Part-2: Choose KPs
        nKP2=0;
        strong1 = pts1;
        selectpts=abs([strong1.Location]); 
        fprintf('\nTotal No.of Chosen Key-Points                    : %d ',length(selectpts));
        
        %% Part-3: Evaluate Features at Selected Points
        multiscaleGridPoints = [SURFPoints(selectpts, 'Scale', 1.6)];
        features = extractFeatures(GI, multiscaleGridPoints,'Upright',true);
        featureMetrics = var(features,[],2); 
        patches=[patches features'];  
        FDCT=patches';

%% Part-5: Forming Clusters
G=[];
numclass=20;
[centers,mincenter,mindist,q2,quality] = FastKmean(FDCT,numclass,1);
for n=1:numclass
    ind= find(mincenter==n);
    for i=1:length(ind)
        G(n,i)= ind(i);
    end
end
% Display Clustered Region
col=jet(size(G,1));m=0;CP=[];cpn=[];
for e=1:size(G,1)
    color=col(e,:);
    for ee=1:size(G,2)
        if G(e,ee)~=0
            k=G(e,ee);
            m=m+1;
            CP(m,:)=abs(selectpts(k,:));
            cpn(m)=e;
        end
    end
end
figure(5),imshow(GI),title('Image with Clustered Key-Points');hold on;
gscatter(CP(:,1),CP(:,2),cpn,[],[],[],'off') 
%% Part-6:  Detect copy-move
RI=GI;
SIM1=[];SIM2=[];
for nG=1:size(G,1) 
    emp=find( G(nG,:)==0);
    if isempty(emp)==0
        A=[];
        for a=1:emp(1)-1
            k=G(nG,a);
            A(a,1:size(FDCT,2))=FDCT(k,:);
            A(a,size(FDCT,2)+1)=k;
        end
    else
        A=[];
        for a=1:size(G,2)
            k=G(nG,a);
            A(a,1:size(FDCT,2))=FDCT(k,:);
            A(a,size(FDCT,2)+1)=k;
        end
    end
    Asorted=RadixSort(A,size(FDCT,2));                                                 
    for i=1:size(Asorted,1)-1
        similar=[];
        for l=1:size(FDCT,2)  %num of features
            s=abs(Asorted(i+1,l)-Asorted(i,l));            
            if s<s_threshold
                similar(l)=1; %similar
            else
                similar(l)=0; %not similar
            end
        end
        if isempty(find(similar==0)) 
            SIM1=[SIM1;Asorted(i,size(FDCT,2)+1)];
            SIM2=[SIM2;Asorted(i+1,size(FDCT,2)+1)];
        end
    end  
end
SIMindex=[SIM1;SIM2];
P1=floor(selectpts(SIM1,:));
P2=floor(selectpts(SIM2,:));
P=[P1;P2];
cpn=[ones(size(P1,1),1);2*ones(size(P1,1),1)];
figure(6),imshow(GI),title('Matched Points before RANSAC');hold on;
gscatter(P(:,1),P(:,2),cpn);
%% Part-7: Hierarchical Agglomerative Clustering
    metric='ward';
    distance_p=pdist(P);
    Z = linkage(distance_p,metric);
    c = cluster(Z,'cutoff',2,'depth',4);
    % show an image depicting clusters and matches
    n_match_before=size(P1,1);
        figure(7),imshow(GI);,title('Matched Points before RANSAC');hold on;
        for i = 1: size(P1,1)
%             line([P1(i,1)' P2(i,1)'], [P1(i,2)' P2(i,2)'], 'Color', 'g');
            line([P1(i,1)' P2(i,1)'], [P1(i,2)' P2(i,2)'], 'Color', 'c');
        end
        % gscatter(P(:,1),P(:,2),c);
        gscatter(P(:,1),P(:,2),c,[],[],[],'off'); 

%% Part-8: Perform RANSAC
RRR=1
if RRR==0
[tformTotal,zz1,zz2] = estimateGeometricTransform(P1,P2,'affine');  % 'similarity', 'affine', or 'projective'
        %figure(8),showMatchedFeatures(GI,GI,zz1,zz2);
        KP=zz1(:,1:2);
        TP=zz2(:,1:2);
        figure(8),imshow(GI);title('Matched Points After RANSAC');,hold on;
        for i = 1: size(KP,1)
%            line([KP(i,1)' TP(i,1)'], [KP(i,2)' TP(i,2)'], 'Color', 'y');  %(b,g,r,c,m,y,black)
            line([KP(i,1)' TP(i,1)'], [KP(i,2)' TP(i,2)'], 'Color', 'g');  %(b,g,r,c,m,y,black)
            
        end
        cpn=[ones(size(KP,1),1);2*ones(size(KP,1),1)]; 
        KK=[KP;TP];
        gscatter(KK(:,1),KK(:,2),cpn,[],[],[],'off'); 
%% Printing Results                           
            fprintf('\nSize of the image:%4d %4d %4d',height,width,numChannels);
            fprintf('\nNo of SURF KPs: %d  \nNo.of FAST KPs             : %d ',nKP1,nKP2);
            fprintf('\nTotal No.of Chosen Key-Points                    : %d ',length(selectpts));
            fprintf('\nNo.of Matching Points Before RANSAC              : %d ',n_match_before);
else
%% Part-8: Perform RANSAC
inliers1 = [];
inliers2 = [];
P1=P1';
P2=P2';
    % given clusters of matched points compute the number of transformations
    num_gt=0;zz1=[];zz2=[];
    c_max = max(c);
    if(c_max > 1)
        n_combination_cluster = nchoosek(1:c_max,2);
        for i=1:1:size(n_combination_cluster,1)
            k=n_combination_cluster(i,1);
            j=n_combination_cluster(i,2);
            z1=[];
            z2=[];
            for r=1:1:size(P1,2)
                if c(r)==k && c(r+size(P1,2))==j
                    z1 = [z1; [P(r,:) 1]];
                    z2 = [z2; [P(r+size(P1,2),:) 1]];
                end
                if c(r)==j && c(r+size(P1,2))==k
                    z1 = [z1; [P(r+size(P1,2),:) 1]];
                    z2 = [z2; [P(r,:) 1]];
                end
            end
            %z1 are coordinates of points in the first cluster 
            %z2 are coordinates of points in the second cluster   
            min_cluster_pts=3;
            if (size(z1,1) > min_cluster_pts && size(z2,1) > min_cluster_pts)
                % run ransacfithomography for affine homography
                
                [H, inliers, dx, dy, xc, yc] = Perform_RANSAC(z1', z2', 0.05);
                
                if size(H,1)==0
                    num_gt = num_gt;
                else
                    H = H / H(3,3);
                    num_gt = num_gt+1;
                    inliers1 = [inliers1; [z1(inliers,1) z1(inliers,2)]];
                    inliers2 = [inliers2; [z2(inliers,1) z2(inliers,2)]];
%                     z1=z1
%                     z2=z2
%                     inliers=inliers
                    show_inliers(name,z1',z2',inliers);
                    zz1=[zz1;z1];zz2=[zz2;z2];
                end
            end
        end  
    end
    inliers=[1:size(zz1,1)];
    show_inliers(name,zz1',zz2',inliers);
            fprintf('\nNo of SURF KPs: %d  \nNo.of FAST KPs             : %d ',nKP1,nKP2);
            fprintf('\nTotal No.of Chosen Key-Points                    : %d ',length(selectpts));
            fprintf('\nNo.of Matching Points Before RANSAC              : %d ',n_match_before);
end
   if zz1
        KP=zz1(:,1:2);
        TP=zz2(:,1:2);
            fprintf('\nForged Image: No. of Matching Points After RANSAC: %d \n',size(KP,1));
            nn=length(KP);
            for i=1:3:nn
                if i+2 <=nn
                    fprintf('\n%4d, %4d:%4d, %4d:%4d, %4d:%4d, %4d:%4d, %4d:%4d, %4d',KP(i,:),TP(i,:),KP(i+1,:),TP(i+1,:),KP(i+2,:),TP(i+2,:));
                elseif i+1<=nn-1
                    fprintf('\n%4d, %4d:%4d, %4d:%4d, %4d:%4d, %4d',KP(i,:),TP(i,:),KP(i+1,:),TP(i+1,:));
                else
                    fprintf('\n%4d, %4d:%4d, %4d',KP(i,:),TP(i,:));
                end
            end
            if i+1<nn
                fprintf('\n%4d, %4d:%4d, %4d:%4d, %4d:%4d, %4d',KP(i,:),TP(i,:),KP(i+1,:),TP(i+1,:));
            else
                fprintf('\n%4d, %4d:%4d, %4d',KP(i,:),TP(i,:));
            end
    else
        fprintf('\n\nOriginal Image without any Forgery');
    end
%% ----------------------------------------- END of MAIN PROGRAM -----------------------------------------

function [centers,mincenter,mindist,q2,quality] = FastKmean(data,initcenters,method)
if nargin < 3 method = 2; end
[n,dim] = size(data);

if max(size(initcenters)) == 1
    k = initcenters;
    [centers, mincenter, mindist, lower, computed] = anchors(mean(data),k,data);
    total = computed;
    skipestep = 1;
else
    centers = initcenters;
    mincenter = zeros(n,1);
    total = 0;
    skipestep = 0;
    [k,dim2] = size(centers);
    if dim ~= dim2 error('dim(data) ~= dim(centers)'); end;
end

nchanged = n;
iteration = 0;
oldmincenter = zeros(n,1);

while nchanged > 0
    % do one E step, then one M step
    computed = 0;
    
    if method == 0 & ~skipestep
        for i = 1:n
            for j = 1:k
                distmat(i,j) = calcdist(data(i,:),centers(j,:));
            end
        end
        [mindist,mincenter] = min(distmat,[],2);
        computed = k*n;
        
    elseif (method == 1 | (method == 2 & iteration == 0)) & ~skipestep
        mindist = Inf*ones(n,1);
        lower = zeros(n,k);
        for j = 1:k
            jdist = calcdist(data,centers(j,:));
            lower(:,j) = jdist;
            track = find(jdist < mindist);
            mindist(track) = jdist(track);
            mincenter(track) = j;
        end
        computed = k*n;
        
    elseif method == 2 & ~skipestep
        computed = 0;
        
        
        nndist = min(centdist,[],2);
        mobile = find(mindist > nndist(mincenter));
        
        mdm = mindist(mobile);
        mcm = mincenter(mobile);
        
        for j = 1:k
            track = find(mdm > centdist(mcm,j));
            if isempty(track) continue; end
            alt = find(mdm(track) > lower(mobile(track),j));
            if isempty(alt) continue; end
            track1 = mobile(track(alt));
            
            
            redo = find(~recalculated(track1));
            redo = track1(redo);
            c = mincenter(redo);
            computed = computed + size(redo,1);
            for jj = unique(c)'
                rp = redo(find(c == jj));
                udist = calcdist(data(rp,:),centers(jj,:));
                lower(rp,jj) = udist;
                mindist(rp) = udist;
            end
            recalculated(redo) = 1;
            
            track2 = find(mindist(track1) > centdist(mincenter(track1),j));
            track1 = track1(track2);
            if isempty(track1) continue; end
            % calculate exact distances to center j
            track4 = find(lower(track1,j) < mindist(track1));
            if isempty(track4) continue; end
            track5 = track1(track4);
            jdist = calcdist(data(track5,:),centers(j,:));
            computed = computed + size(track5,1);
            lower(track5,j) = jdist;
            track2 = find(jdist < mindist(track5));
            track3 = track5(track2);
            mindist(track3) = jdist(track2);
            mincenter(track3) = j;
        end % for j=1:k
    end % if method
    oldcenters = centers;
    diff = find(mincenter ~= oldmincenter);
    diffj = unique([mincenter(diff);oldmincenter(diff)])';
    diffj = diffj(find(diffj > 0));
    if size(diff,1) < n/3 & iteration > 0
        for j = diffj
            plus = find(mincenter(diff) == j);
            minus = find(oldmincenter(diff) == j);
            oldpop = pop(j);
            pop(j) = pop(j) + size(plus,1) - size(minus,1);
            if pop(j) == 0 continue; end
            centers(j,:) = (centers(j,:)*oldpop + sum(data(diff(plus),:),1) - sum(data(diff(minus),:),1))/pop(j);
        end
    else
        for j = diffj
            track = find(mincenter == j);
            pop(j) = size(track,1);
            if pop(j) == 0 continue; end
            % it's correct to have mean(data(track,:),1) but this can make answer worse!
            centers(j,:) = mean(data(track,:),1);
        end
    end
    if method == 2
        for j = diffj
            offset = calcdist(centers(j,:),oldcenters(j,:));
            computed = computed + 1;
            if offset == 0 continue; end
            track = find(mincenter == j);
            mindist(track) = mindist(track) + offset;
            lower(:,j) = max(lower(:,j) - offset,0);
        end
        
        % compute distance between each pair of centers
        % modify centdist to make "find" using it faster
        recalculated = zeros(n,1);
        realdist = alldist(centers);
        centdist = 0.5*realdist + diag(Inf*ones(k,1));
        computed = computed + k + k*(k-1)/2;
    end
    nchanged = size(diff,1) + skipestep;
    iteration = iteration+1;
    skipestep = 0;
    oldmincenter = mincenter;
    total = total + computed;
end % while nchanged > 0
udist = calcdist(data,centers(mincenter,:));
quality = mean(udist);
q2 = mean(udist.^2);
end
end

function [centers, mincenter, mindist, lower, computed] = anchors(firstcenter,k,data)
% choose k centers by the furthest-first method

[n,dim] = size(data);
centers = zeros(k,dim);
lower = zeros(n,k);
mindist = Inf*ones(n,1);
mincenter = ones(n,1);
computed = 0;
centdist = zeros(k,k);

for j = 1:k
    if j == 1
        newcenter = firstcenter;
    else
        [maxradius,i] = max(mindist);
        newcenter = data(i,:);
    end

    centers(j,:) = newcenter;
    centdist(1:j-1,j) = calcdist(centers(1:j-1,:),newcenter);
    centdist(j,1:j-1) = centdist(1:j-1,j)';
    computed = computed + j-1;
    
    inplay = find(mindist > centdist(mincenter,j)/2);
    newdist = calcdist(data(inplay,:),newcenter);
    computed = computed + size(inplay,1);
    lower(inplay,j) = newdist;
        
%    other = find(mindist <= centdist(mincenter,j)/2);
%    if ~isempty(other)
%        lower(other,j) = centdist(mincenter(other),j) - mindist(other);
%    end    
        
    move = find(newdist < mindist(inplay));
    shift = inplay(move);
    mincenter(shift) = j;
    mindist(shift) = newdist(move);
end
end

function distances = calcdist(data,center)
%  input: vector of data points, single center or multiple centers
% output: vector of distances

[n,dim] = size(data);
[n2,dim2] = size(center);

% Using repmat is slower than using ones(n,1)
%   delta = data - repmat(center,n,1);
%   delta = data - center(ones(n,1),:);
% The following is fastest: not duplicating the center at all

if n2 == 1
    distances = sum(data.^2, 2) - 2*data*center' + center*center';
elseif n2 == n
    distances = sum( (data - center).^2 ,2);
else
    error('bad number of centers');
end

% Euclidean 2-norm distance:
distances = sqrt(distances);
end

function Asorted=RadixSort(A,digit)
i=digit;
Asorted=A;
while i>=1
    [x,inx]=sort(Asorted(:,i));
    Asorted(:,:)=Asorted(inx(:),:);
    i=i-1;
end
end %function

% NORMALISE2DPTS - normalises 2D homogeneous points
%
% Function translates and normalises a set of 2D homogeneous points 
% so that their centroid is at the origin and their mean distance from 
% the origin is sqrt(2).  This process typically improves the
% conditioning of any equations used to solve homographies, fundamental
% matrices etc.
%
% Usage:   [newpts, T] = normalise2dpts(pts)
%
% Argument:
%   pts -  3xN array of 2D homogeneous coordinates
%
% Returns:
%   newpts -  3xN array of transformed 2D homogeneous coordinates.  The
%             scaling parameter is normalised to 1 unless the point is at
%             infinity. 
%   T      -  The 3x3 transformation matrix, newpts = T*pts
%           
% If there are some points at infinity the normalisation transform
% is calculated using just the finite points.  Being a scaling and
% translating transform this will not affect the points at infinity.

% Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% May 2003      - Original version
% February 2004 - Modified to deal with points at infinity.
% December 2008 - meandist calculation modified to work with Octave 3.0.1
%                 (thanks to Ron Parr)
%
% October 2009  - modified by I. Amerini and G. Serra; now the function 
%                 returns the centroid c of the set of points pts

function [newpts, T, c] = normalise2dpts2(pts)

    if size(pts,1) ~= 3
        error('pts must be 3xN');
    end
    
    % Find the indices of the points that are not at infinity
    finiteind = find(abs(pts(3,:)) > eps);
    
    if length(finiteind) ~= size(pts,2)
        warning('Some points are at infinity');
    end
    
    % For the finite points ensure homogeneous coords have scale of 1
    pts(1,finiteind) = pts(1,finiteind)./pts(3,finiteind);
    pts(2,finiteind) = pts(2,finiteind)./pts(3,finiteind);
    pts(3,finiteind) = 1;
    
    c = mean(pts(1:2,finiteind)')';           % Centroid of finite points
    newp(1,finiteind) = pts(1,finiteind)-c(1); % Shift origin to centroid.
    newp(2,finiteind) = pts(2,finiteind)-c(2);
    
    dist = sqrt(newp(1,finiteind).^2 + newp(2,finiteind).^2);
    meandist = mean(dist(:));  % Ensure dist is a column vector for Octave 3.0.1
    
    if (meandist~=0)
        scale = sqrt(2)/meandist;
    else
        scale=1;
    end
    
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    
    newpts = T*pts;
end  

function show_inliers(imagefile, zz1, zz2, inliers)
image1 = rgb2gray(imread(imagefile));
if zz1
    figure(8);
    imshow(image1);title('Forged Image with Matched Points after RANSAC');hold on;
    for i=1:1:size(inliers,2)
        line([zz1(1,inliers(1,i)) zz2(1,inliers(1,i))], ...
              [zz1(2,inliers(1,i)) zz2(2,inliers(1,i))], 'Color', 'g');
    end
    zz1=zz1';
    zz2=zz2';
    zz2(:,3)=2;
    PP=[zz1(:,1:2);zz2(:,1:2)];
    CC=[zz1(:,3);zz2(:,3)];
    gscatter(PP(:,1),PP(:,2),CC);
else
    figure(9);
    imshow(image1);title('Original Image without any Matched Points after RANSAC');
end
end

