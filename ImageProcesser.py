import cv2 as cv
import numpy as np
import json

def find_transformation(ref, input_frame, scale_factor=1, MIN_MATCH_COUNT=10, draw=False, printout=False):
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

    cv.imshow('Matches', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    img1 = cv.imread('app/public/opencv_frame_0.png', 0)  # 180 theta
    img2 = cv.imread('app/public/opencv_frame_1.png', 0)

    # Compare test image with img1
    print(f"Comparing image 1 with image 2:")
    result1 = find_transformation(img1, img2, scale_factor=1, draw=True, printout=True)
        
    if result1:
        dx1, dy1, theta1 = result1
        print(f"Results for comparison with image 1:")
        print(f"Displacement (dx, dy): ({dx1:.2f}, {dy1:.2f})")
        print(f"Rotation angle (theta): {theta1:.2f} degrees")

        # Create the JSON object with x and y
        json_data = {
            "x": round(dx1, 2),
            "y": round(dy1, 2),
            "theta": round(theta1, 2)
        }

        # Write the JSON to a file
        with open("data.json", "w") as out_file:
            json.dump(json_data, out_file, indent=4)
        
        print("JSON data saved to 'data.json':", json_data)
    else:
        print(f"Could not process images")
