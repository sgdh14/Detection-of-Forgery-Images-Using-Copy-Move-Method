function [H, inliers, dx, dy, cx, cy] = Perform_RANSAC(x1, x2, t)

    if ~all(size(x1)==size(x2))
        error('Data sets x1 and x2 must have the same dimension');
    end
    
    [rows,npts] = size(x1);
    if rows~=2 && rows~=3
        error('x1 and x2 must have 2 or 3 rows');
    end
    
    if npts < 3
        error('Must have at least 3 points to fit homography');
    end
    
    if rows == 2    % Pad data with homogeneous scale factor of 1
        x1 = [x1; ones(1,npts)];
        x2 = [x2; ones(1,npts)];        
    end

    % Normalise each set of points so that the origin is at centroid and
    % mean distance from origin is sqrt(2).  normalise2dpts also ensures the
    % scale parameter is 1.  Note that 'homography2d' will also call
    % 'normalise2dpts' but the code in 'ransac' that calls the distance
    % function will not - so it is best that we normalise beforehand.
    [x1, T1, c1] = normalise2dpts2(x1);
    [x2, T2, c2] = normalise2dpts2(x2);
    
    % c1 and c2 are centers of the clusters
    cx = c1(1);
    cy = c1(2);
    dx = c2(1)-c1(1);
    dy = c2(2)-c1(2);
    s = 4;  % Minimum No of points needed to fit a homography.
    
    fittingfn = @homography2d;
    distfn    = @homogdist2d;
    degenfn   = @isdegenerate;
    % x1 and x2 are 'stacked' to create a 6xN array for ransac
    [H, inliers] = ransac([x1; x2], fittingfn, distfn, degenfn, s, t);
    
    % Now do a final least squares fit on the data points considered to
    % be inliers.
    %H = homography2d(x1(:,inliers), x2(:,inliers));
    
    if size(x1(:,inliers),2)>3 && size(H,1)>0
        H = vgg_Haffine_from_x_MLE(x1(:,inliers),x2(:,inliers));
        % Denormalise
        H = T2\H*T1;  
    end
end
 

%----------------------------------------------------------------------
% Function to evaluate the symmetric transfer error of a homography with
% respect to a set of matched points as needed by RANSAC.

function [inliers, H] = homogdist2d(H, x, t)
    
    x1 = x(1:3,:);   % Extract x1 and x2 from x
    x2 = x(4:6,:);    
    
    % Calculate, in both directions, the transfered points    
    Hx1    = H*x1;
    invHx2 = H\x2;
    
    % Normalise so that the homogeneous scale parameter for all coordinates
    % is 1.
    
    x1     = hnormalise(x1);
    x2     = hnormalise(x2);     
    Hx1    = hnormalise(Hx1);
    invHx2 = hnormalise(invHx2); 
    
    d2 = sum((x1-invHx2).^2)  + sum((x2-Hx1).^2);
    inliers = find(abs(d2) < t); 
end
    
    
%----------------------------------------------------------------------
% Function to determine if a set of 4 pairs of matched  points give rise
% to a degeneracy in the calculation of a homography as needed by RANSAC.
% This involves testing whether any 3 of the 4 points in each set is
% colinear. 
     
function r = isdegenerate(x)

    x1 = x(1:3,:);    % Extract x1 and x2 from x
    x2 = x(4:6,:);    
    
    r = ...
    iscolinear(x1(:,1),x1(:,2),x1(:,3)) | ...
    iscolinear(x1(:,1),x1(:,2),x1(:,4)) | ...
    iscolinear(x1(:,1),x1(:,3),x1(:,4)) | ...
    iscolinear(x1(:,2),x1(:,3),x1(:,4)) | ...
    iscolinear(x2(:,1),x2(:,2),x2(:,3)) | ...
    iscolinear(x2(:,1),x2(:,2),x2(:,4)) | ...
    iscolinear(x2(:,1),x2(:,3),x2(:,4)) | ...
    iscolinear(x2(:,2),x2(:,3),x2(:,4));
end
    
function [M, inliers] = ransac(x, fittingfn, distfn, degenfn, s, t, feedback, ...
                               maxDataTrials, maxTrials)

    Octave = exist('OCTAVE_VERSION') ~= 0;

    % Test number of parameters
    error ( nargchk ( 6, 9, nargin ) );
    
    if nargin < 9; maxTrials = 1000;    end; 
    if nargin < 8; maxDataTrials = 100; end; 
    if nargin < 7; feedback = 0;        end;
    
    [rows, npts] = size(x);                 
    
    p = 0.99;         % Desired probability of choosing at least one sample
                      % free from outliers

    bestM = NaN;      % Sentinel value allowing detection of solution failure.
    trialcount = 0;
    bestscore =  0;    
    N = 1;            % Dummy initialisation for number of trials.
    
    while N > trialcount
        
        % Select at random s datapoints to form a trial model, M.
        % In selecting these points we have to check that they are not in
        % a degenerate configuration.
        degenerate = 1;
        count = 1;
        %follia delle 9.40 - Beppe 
        M=feval(fittingfn, x(:,:));
        while degenerate
            % Generate s random indicies in the range 1..npts
            % (If you do not have the statistics toolbox, or are using Octave,
            % use the function RANDOMSAMPLE from my webpage)
	    if Octave
		ind = randomsample(npts, s);
	    else
		ind = randsample(npts, s);
	    end

            % Test that these points are not a degenerate configuration.
            degenerate = feval(degenfn, x(:,ind));
            
            if ~degenerate 
                % Fit model to this random selection of data points.
                % Note that M may represent a set of models that fit the data in
                % this case M will be a cell array of models
                M = feval(fittingfn, x(:,ind));
                
                % Depending on your problem it might be that the only way you
                % can determine whether a data set is degenerate or not is to
                % try to fit a model and see if it succeeds.  If it fails we
                % reset degenerate to true.
                if isempty(M)
                    degenerate = 1;
                end
            end
            
            % Safeguard against being stuck in this loop forever
            count = count + 1;
            if count > maxDataTrials
                warning('Unable to select a nondegenerate data set');
                break
            end
        end
        
        % Once we are out here we should have some kind of model...        
        % Evaluate distances between points and model returning the indices
        % of elements in x that are inliers.  Additionally, if M is a cell
        % array of possible models 'distfn' will return the model that has
        % the most inliers.  After this call M will be a non-cell object
        % representing only one model.
        [inliers, M] = feval(distfn, M, x, t);
        
        % Find the number of inliers to this model.
        ninliers = length(inliers);
        
        if ninliers > bestscore    % Largest set of inliers so far...
            bestscore = ninliers;  % Record data for this model
            bestinliers = inliers;
            bestM = M;
            
            % Update estimate of N, the number of trials to ensure we pick, 
            % with probability p, a data set with no outliers.
            fracinliers =  ninliers/npts;
            pNoOutliers = 1 -  fracinliers^s;
            pNoOutliers = max(eps, pNoOutliers);  % Avoid division by -Inf
            pNoOutliers = min(1-eps, pNoOutliers);% Avoid division by 0.
            N = log(1-p)/log(pNoOutliers);
        end
        
        trialcount = trialcount+1;
        if feedback
            fprintf('trial %d out of %d         \r',trialcount, ceil(N));
        end

        % Safeguard against being stuck in this loop forever
        if trialcount > maxTrials
            warning( ...
            sprintf('ransac reached the maximum number of %d trials',...
                    maxTrials));
            break
        end     
    end
    fprintf('\n');
    
    if ~isnan(bestM)   % We got a solution 
        M = bestM;
        inliers = bestinliers;
    else           
        M = [];
        inliers = [];
        warning('ransac was unable to find a useful solution');
    end
end

function H = homography2d(varargin)
    
    [x1, x2] = checkargs(varargin(:));

    % Attempt to normalise each set of points so that the origin 
    % is at centroid and mean distance from origin is sqrt(2).
    [x1, T1] = normalise2dpts2(x1);
    [x2, T2] = normalise2dpts2(x2);
    
    % Note that it may have not been possible to normalise
    % the points if one was at infinity so the following does not
    % assume that scale parameter w = 1.
    
    Npts = length(x1);
    A = zeros(3*Npts,9);
    
    O = [0 0 0];
    for n = 1:Npts
	X = x1(:,n)';
	x = x2(1,n); y = x2(2,n); w = x2(3,n);
	A(3*n-2,:) = [  O  -w*X  y*X];
	A(3*n-1,:) = [ w*X   O  -x*X];
	A(3*n  ,:) = [-y*X  x*X   O ];
    end
    
    [U,D,V] = svd(A,0); % 'Economy' decomposition for speed
    
    % Extract homography
    H = reshape(V(:,9),3,3)';
    
    % Denormalise
    H = T2\H*T1;
end   

%--------------------------------------------------------------------------
% Function to check argument values and set defaults

function [x1, x2] = checkargs(arg);
    
    if length(arg) == 2
	x1 = arg{1};
	x2 = arg{2};
	if ~all(size(x1)==size(x2))
	    error('x1 and x2 must have the same size');
	elseif size(x1,1) ~= 3
	    error('x1 and x2 must be 3xN');
	end
	
    elseif length(arg) == 1
	if size(arg{1},1) ~= 6
	    error('Single argument x must be 6xN');
	else
	    x1 = arg{1}(1:3,:);
	    x2 = arg{1}(4:6,:);
	end
    else
	error('Wrong number of arguments supplied');
    end
end

function nx = hnormalise(x)
    
    [rows,npts] = size(x);
    nx = x;

    % Find the indices of the points that are not at infinity
    finiteind = find(abs(x(rows,:)) > eps);

    if length(finiteind) ~= npts
        warning('Some points are at infinity');
    end

    % Normalise points not at infinity
    for r = 1:rows-1
	nx(r,finiteind) = x(r,finiteind)./x(rows,finiteind);
    end
    nx(rows,finiteind) = 1;
end   


function r = iscolinear(p1, p2, p3, flag)
    if nargin == 3   % Assume inhomogeneous coords
	flag = 'inhomog';
    end
    
    if ~all(size(p1)==size(p2)) | ~all(size(p1)==size(p3)) | ...
        ~(length(p1)==2 | length(p1)==3)                              
        error('points must have the same dimension of 2 or 3');
    end
    
    % If data is 2D, assume they are 2D inhomogeneous coords. Make them
    % homogeneous with scale 1.
    if length(p1) == 2    
        p1(3) = 1; p2(3) = 1; p3(3) = 1;
    end

    if flag(1) == 'h'
	% Apply test that allows for homogeneous coords with arbitrary
        % scale.  p1 X p2 generates a normal vector to plane defined by
        % origin, p1 and p2.  If the dot product of this normal with p3
        % is zero then p3 also lies in the plane, hence co-linear.
	r =  abs(dot(cross(p1, p2),p3)) < eps;
    else
	% Assume inhomogeneous coords, or homogeneous coords with equal
        % scale.
	r =  norm(cross(p2-p1, p3-p1)) < eps;
    end
end

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

function pc = vgg_condition_2d(p,C)
[r,c] = size(p);
if r == 2
  pc = vgg_get_nonhomg(C * vgg_get_homg(p));
elseif r == 3
  pc = C * p;
else
  error ('rows != 2 or 3');
end
end

function x = vgg_get_nonhomg(x)
% p = vgg_get_nonhomg(h)
% Convert a set of homogeneous points to non-homogeneous form
% Points are stored as column vectors, stacked horizontally, e.g.
%  [x0 x1 x2 ... xn ;
%   y0 y1 y2 ... yn ;
%   w0 w1 w2 ... wn ]
% Modified by TW
if isempty(x)
  x = []; 
  return; 
end
d = size(x,1) - 1;
x = x(1:d,:)./(ones(d,1)*x(end,:));
return
end

function H = vgg_Haffine_from_x_MLE(xs1,xs2)
% H = vgg_Haffine_from_x_MLE(xs1,xs2)
%
% Compute MLE for affine H, i.e. find H and xhat1 such that
% d^2(xs1,xhat1) + d^2(xs2,xhat2) minimized where xhat2 is affine transf of xhat1.
%
% Parameters:
%   xs1, xs2 ... double(3,N), N pairs of corresponding points (homogeneous)
%   H ... double(3,3), affine transformation
%
% See HZ page 115 1st edition, page 130 2nd edition
% az 17/11/2001

if any(size(xs1) ~= size(xs2))
 error ('Input point sets are different sizes!')
end

% condition points

nonhomg = vgg_get_nonhomg(xs1);
means = mean(nonhomg');
maxstds = max(std(nonhomg'));
C1 = diag([1/maxstds 1/maxstds 1]);  % only similarity 
C1(:,3) = [-means/maxstds 1]';

nonhomg = vgg_get_nonhomg(xs2);
means = mean(nonhomg');
C2 = C1;            % nb must use same scaling for both point sets
C2(:,3) = [-means/maxstds 1]';

xs1 = vgg_condition_2d(xs1,C1);
xs2 = vgg_condition_2d(xs2,C2);

% NB conditioned points have mean zero, so translation
% part of affine transf is zero 2-vector

xs1nh = vgg_get_nonhomg(xs1);
xs2nh = vgg_get_nonhomg(xs2);

A = [xs1nh;xs2nh]';

% Extract nullspace
[u,s,v] = svd(A); 
s = diag(s);
 
nullspace_dimension = sum(s < eps * s(2) * 1e3);
if nullspace_dimension > 2
  fprintf('Nullspace is a bit roomy...');
end

% compute affine matrix from two largest singular vecs

B = v(1:2,1:2);
C = v(3:4,1:2);

H = [ C * pinv(B) , zeros(2,1); 0 0 1];

% decondition
H = inv(C2) * H * C1;

H = H/H(3,3);
end