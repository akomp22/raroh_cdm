            ret, frame = cam.get_frame()
            if not ret:
                print("Error reading frame")
                break

            coord, mask_cleaned = find_red_spot_center(frame)
            print(coord)