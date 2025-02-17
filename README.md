Introduction


• Proliferation of digital manipulation tools has increased the risk of image tampering. Copy-move forgery involves duplicating a portion of an image to conceal or modify content.
• Detecting such forgeries in medical images is imperative for maintaining diagnostic trustworthiness and ensuring patient safety. The proposed method leverages advanced image processing and machine learning algorithms to identify duplicated regions within medical images.
• Experimental results demonstrate the effectiveness of this approach across various medical imaging modalities (X-rays, MRIs, CT scans).
• This method offers a valuable tool for forensic analysis and quality assurance in medical imaging.

Architecture
• Image authentication begins with a database and a test image.
• Test image undergoes RGB to grayscale conversion and noise removal.
• Key interest points are detected using SURF (Speeded-Up Robust Features).
• Features are reduced using Principal Component Analysis (PCA).
• Reduced features are clustered using K-Means clustering.
• Feature matching is conducted to compare test image features with database images.
• False matches are removed using RANSAC (Random Sample Consensus).
• System classifies the image as either authentic or tampered.

Tools and Technologies Used
• MATLAB: Utilized for implementing the image processing and machine learning algorithms.
• SURF: For key interest point detection.
• PCA: For feature reduction.
• K-Means Clustering: For clustering similar features.
• RANSAC: For removing false matches.

Key Achievements
• Developed a robust method for detecting copy-move forgeries in digital images.
• Demonstrated effectiveness across multiple types of images.
• Contributed to enhancing reliability and trust in digital imaging practices.
