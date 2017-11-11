for file in ./*.JPG; do
  convert "$file" -rotate 90 "${file%.JPG}"_rotated.JPG
done
