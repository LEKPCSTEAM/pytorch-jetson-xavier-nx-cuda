import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 (เลือก model ตามที่ต้องการ เช่น yolov8n.pt หรือ yolov8s.pt)
model = YOLO('yolov8n.pt')  # ใช้รุ่น nano เล็กสุด, เปลี่ยนเป็น s/m/l/x ได้

# เปิดกล้องเว็บแคมหรือใส่ path วิดีโอ/ภาพ
cap = cv2.VideoCapture(2)  # เปลี่ยนเป็น 'video.mp4' หรือ 'img.jpg' ได้

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # รัน YOLO ตรวจจับวัตถุ
    results = model(frame)

    # วาดกรอบและ label ด้วยฟังก์ชัน .plot()
    annotated_frame = results[0].plot()  # plot() คืน numpy array ที่มีกรอบและ label แล้ว

    # แสดงผลลัพธ์
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()

