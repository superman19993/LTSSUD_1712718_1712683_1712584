# LTSSUD_1712718_1712683_1712584

## Lập trình song song ứng dung.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1YkQHvIPv5-THZ0ETgPHOIy2O2teKBN)
###Giảng viên: Thầy Trần Trung Kiên

## Thông tin nhóm
**Nhóm 4**:
1. 1712718: Huỳnh Thanh Sang.
2. 1712683: Phạm Hoàng Phương.
3. 1712584: Nguyễn Công Lý


* Trong đồ án này, nhóm lựa chọn tối ưu hoá cho 1 model Object Detection & Segmentation dựa vào Mask RCNN đã được huấn luyện sẵn.
 
**Tài liệu nhóm**:
* Link Colab thực hiện: [Link Colab](https://colab.research.google.com/drive/1V1YkQHvIPv5-THZ0ETgPHOIy2O2teKBN)
* Kế hoạch phân công công việc của nhóm [tại đây](https://docs.google.com/spreadsheets/d/1CDZhYaKv_k68HpzkTHc-RwWgaBjBY12RCkK1squYVBM/edit?usp=sharing).

## Bài toán:
Đề tài: Nhật diện và phân đoạn đối tượng với Mask-RCNN
Input:
* Ảnh đầu vào có kích thước W x H

Output:
* Các bounding box:[cx, cy, w, h] (tọa độ tâm + kích thước) tại các vị trí nghi ngờ có cá thể

Ý nghĩa thực tế: 

Lý do chọn đề tài: 
* Bài toán sử dụng mạng CNN (liên quan đến tác vụ convolution trên ảnh)

