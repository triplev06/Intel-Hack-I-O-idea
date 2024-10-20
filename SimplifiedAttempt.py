import cv2 as cv
import numpy as np

def find_transformation(ref, input_frame, scale_factor=0.50, MIN_MATCH_COUNT=10, draw=False, printout=False):
    img1 = cv.resize(ref, None, fx=scale_factor, fy=scale_factor)
    img2 = cv.resize(input_frame, None, fx=scale_factor, fy=scale_factor)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        rect_dst = np.int32(dst)
        h2, w2 = img2.shape
        dx = w2 // 2 - (rect_dst[0][0][0] + rect_dst[2][0][0]) // 2
        dy = h2 // 2 - (rect_dst[0][0][1] + rect_dst[2][0][1]) // 2
        theta = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        if printout:
            print(f"Displacement (dx, dy): ({dx/scale_factor}, {dy/scale_factor})")
            print(f"Rotation angle (theta): {theta} degrees")

        if draw:
            draw_matches(img1, kp1, img2, kp2, good, mask)

        return dx/scale_factor, dy/scale_factor, theta
    else:
        print(f"Not enough matches found - {len(good)}/{MIN_MATCH_COUNT}")
        return None

def draw_matches(img1, kp1, img2, kp2, good, mask):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=mask.ravel().tolist(),
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    for i, match in enumerate(good):
        if mask[i]:
            x1, y1 = map(int, kp1[match.queryIdx].pt)
            x2, y2 = map(int, kp2[match.trainIdx].pt)
            #cv.putText(img3, f"({x1}, {y1})", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            #cv.putText(img3, f"({x2}, {y2})", (x2 + img1.shape[1], y2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv.imshow('Matches', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Load the two reference images
    img1 = cv.imread('snakesmall1.jpg', 0)  # 180 theta
    img2 = cv.imread('snakesmall2.jpg', 0)

    # Compare test image with img1
    print(f"Comparing snakesmall2.jpg with snakesmall1.jpg:")
    result1 = find_transformation(img1, img2, scale_factor=0.50, draw=True, printout=True)
        
    if result1:
            dx1, dy1, theta1 = result1
            print(f"Results for comparison with snakesmall1.jpg:")
            print(f"Displacement (dx, dy): ({dx1:.2f}, {dy1:.2f})")
            print(f"Rotation angle (theta): {theta1:.2f} degrees")
    else:
            print(f"Could not process snakesmall2.jpg with snakesmall1.jpg")
        
    print("---")
        
        # Compare test image with img2
    print(f"Comparing snakesmall2.jpg with snakesmall2.jpg:")
    result2 = find_transformation(img2, img2, scale_factor=0.50, draw=True, printout=True)
        
    if result2:
            dx2, dy2, theta2 = result2
            print(f"Results for comparison with snakesmall2.jpg:")
            print(f"Displacement (dx, dy): ({dx2:.2f}, {dy2:.2f})")
            print(f"Rotation angle (theta): {theta2:.2f} degrees")
    else:
            print(f"Could not process snakesmall2.jpg with snakesmall2.jpg")
        
    print("=" * 30)

    def load_image(result2):
            img = cv.imread(result2, 0)
    if img is None:
        print(f"Error: Unable to load image snakesmall2.jpg")
    
