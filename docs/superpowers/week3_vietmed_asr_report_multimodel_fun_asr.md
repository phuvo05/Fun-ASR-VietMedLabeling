# Báo cáo tuần 3: Pseudo Labeling, Multi-Model Agreement và Quality Scoring cho dữ liệu VietMed ASR

## 1. Tóm tắt kết quả tuần 3

Trong tuần 3, pipeline gán nhãn tự động cho dữ liệu VietMed đã được triển khai theo hướng **pseudo labeling**: sử dụng mô hình ASR mạnh để sinh transcript, kết hợp forced alignment và word-level confidence để đánh giá chất lượng nhãn. Mục tiêu chính là giảm tối đa khối lượng manual labeling nhưng vẫn tạo được dữ liệu đủ tin cậy để phục vụ bước fine-tune mô hình ASR.

Các kết quả chính đạt được:

| Hạng mục | Kết quả |
|---|---:|
| Split đánh giá | `VietMed_unlabeled_000_050.json` |
| Tổng số audio trong split | 7,975 |
| Audio đã có transcript và timestamp | 475 |
| Audio chưa label | 7,500 |
| Tiến độ labeling | 5.96% |
| Tổng số word/timestamp đã sinh | 35,546 |
| Mean confidence | 0.9755 |
| Median confidence | 1.0000 |
| Word confidence `< 0.70` tính cả null | 1,436 |
| Tỷ lệ word cần review | 4.04% |
| Dataset score sơ bộ trên phần đã label | **9.37/10** |
| Multi-model agreement | Đang triển khai với FunASR làm model thứ hai |

Kết quả này cho thấy hướng pseudo labeling bằng ASR kết hợp confidence score là khả thi. Phần dữ liệu đã label có confidence cao, số word cần review chỉ chiếm khoảng 4.04%, giúp giảm đáng kể khối lượng kiểm tra thủ công so với việc nghe lại toàn bộ audio.

---

## 2. Mục tiêu bài toán

Bài toán yêu cầu gán nhãn cho tập audio chưa có transcript trong bộ dữ liệu VietMed:

- **Tập dữ liệu cần label:** `unlabeled_medical_data`
- **Folder dữ liệu:** https://drive.google.com/drive/folders/1Wrofm2FkjYngpxR1ysmkm0XvFiGEvyFM
- **Notebook thực hiện:** https://colab.research.google.com/drive/1izdKoJ0VQJE4DVTvlVE0nkEKEUvy672e?usp=sharing

Yêu cầu chính gồm hai phần:

1. Sử dụng phương pháp tự động hoặc bán tự động để gán nhãn audio, hạn chế manual labeling nhiều nhất có thể.
2. Xây dựng phương pháp đánh giá định lượng chất lượng cặp audio-transcript, sao cho mỗi audio hoặc mỗi dataset có thể được quy đổi thành một score trên thang 0-10.

Định hướng của tuần 3 là xây dựng pipeline pseudo labeling có thể chạy lặp lại, có checkpoint, có transcript, có timestamp, có confidence theo word và có score định lượng để chọn lọc dữ liệu đưa vào fine-tune.

---

## 3. Phương pháp tổng quan

Pipeline tuần 3 được thiết kế theo luồng sau:

```text
Audio VietMed
    ↓
Chuẩn hóa audio
    ↓
Qwen3-ASR-1.7B sinh transcript
    ↓
Qwen3-ForcedAligner-0.6B sinh timestamp theo word
    ↓
Tính confidence theo token và gom về word
    ↓
Lưu pseudo label vào JSON
    ↓
Tính score chất lượng audio-transcript
    ↓
Chia nhóm: dùng trực tiếp / review chọn lọc / chạy lại model khác
```

Điểm quan trọng của pipeline là không chỉ tạo transcript, mà còn tạo thêm metadata phục vụ kiểm soát chất lượng. Mỗi word trong transcript có thể đi kèm `confidence`, `start` và `end`, nhờ đó có thể phát hiện các vùng rủi ro thấp chất lượng để review có chọn lọc.

---

## 4. Mô hình và cấu hình sử dụng

Notebook hiện tại sử dụng mô hình chính là **Qwen3-ASR-1.7B** để sinh transcript tiếng Việt và **Qwen3-ForcedAligner-0.6B** để lấy timestamp theo word.

```python
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
USE_FORCED_ALIGNER = True
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE = "Vietnamese"
MAX_INFERENCE_BATCH_SIZE = 4
```

Ý nghĩa cấu hình:

- `Qwen/Qwen3-ASR-1.7B`: mô hình ASR chính dùng để sinh transcript.
- `Qwen/Qwen3-ForcedAligner-0.6B`: mô hình aligner dùng để sinh timestamp theo word.
- `LANGUAGE = "Vietnamese"`: ép ngôn ngữ đầu ra là tiếng Việt, giảm rủi ro auto-detect sai ngôn ngữ.
- `MAX_INFERENCE_BATCH_SIZE = 4`: batch size vừa phải để phù hợp môi trường Colab GPU.

Pipeline cũng có cơ chế tự chọn kiểu tính toán theo GPU:

- Ưu tiên `bfloat16` nếu GPU hỗ trợ tốt.
- Fallback về `float16` nếu cần tiết kiệm VRAM.
- Nếu `flash-attn3` không khả dụng thì fallback sang `sdpa`.

Cách cấu hình này giúp notebook dễ chạy hơn trên môi trường Colab, đồng thời vẫn tận dụng GPU để tăng tốc inference.

### 4.1. Multi-model agreement đang triển khai với FunASR

Bên cạnh mô hình chính Qwen3-ASR-1.7B, pipeline đang được mở rộng theo hướng **multi-model agreement** với **FunASR** làm mô hình ASR thứ hai. Mục tiêu của bước này là không chỉ dựa vào confidence nội bộ của một mô hình, mà còn dùng mức độ đồng thuận giữa hai transcript để tăng độ tin cậy của pseudo label.

Luồng xử lý mở rộng:

```text
Audio VietMed
    ↓
Qwen3-ASR sinh transcript A + confidence + timestamp
    ↓
FunASR sinh transcript B
    ↓
So sánh transcript A và B bằng CER/WER hoặc similarity
    ↓
Kết hợp confidence + agreement + timestamp validity
    ↓
Chia nhóm: auto accept / review chọn lọc / chạy lại model khác
```

Cách sử dụng agreement:

- Nếu Qwen3-ASR có confidence cao và transcript của FunASR gần giống, audio được xem là pseudo label chất lượng cao.
- Nếu confidence cao nhưng hai model lệch nhau nhiều, audio được đưa vào nhóm cần review vì có khả năng model đang “tự tin sai”.
- Nếu confidence thấp nhưng FunASR cho transcript tương đồng, có thể giữ lại sau khi review các word rủi ro.
- Nếu cả confidence thấp và agreement thấp, audio nên được chạy lại hoặc kiểm tra thủ công.

Việc thêm FunASR giúp pipeline phù hợp hơn với yêu cầu “gán nhãn chất lượng tốt nhất có thể” vì chất lượng nhãn không còn chỉ dựa trên một mô hình duy nhất.

---

## 5. Tiền xử lý audio

Trước khi đưa audio vào mô hình, notebook thực hiện chuẩn hóa bằng hàm `normalize_audio()`. Bước này xử lý các trường hợp phổ biến trong dữ liệu audio thực tế:

- Audio dạng integer được scale về khoảng `[-1, 1]`.
- Audio dạng float nhưng biên độ vượt quá 1 được chuẩn hóa lại.
- Audio stereo hoặc multi-channel được chuyển về mono bằng cách lấy trung bình các channel.
- Biên độ cuối cùng được clip trong khoảng `[-1, 1]`.

Ngoài ra, để tính logits và confidence ổn định, audio được đưa về sample rate 16 kHz:

```python
def read_audio_16k(audio_path):
    wav, sr = read_audio_as_tuple(audio_path)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)
```

Điều này cần thiết vì dữ liệu VietMed hiện có nhiều audio 8 kHz, trong khi nhiều mô hình ASR hoạt động ổn định hơn khi input được chuẩn hóa về 16 kHz.

---

## 6. Pipeline pseudo labeling

Pipeline pseudo labeling gồm các bước chính:

1. Mount Google Drive để đọc audio và file JSON label.
2. Load mô hình ASR và forced aligner.
3. Kiểm tra từng item trong JSON để xác định mẫu nào chưa có `text` hoặc `timestamps`.
4. Gọi endpoint `/transcribe` để sinh transcript, timestamp và confidence.
5. Ghi kết quả trở lại file JSON label.
6. Bỏ qua các mẫu đã có nhãn để có thể resume khi notebook bị ngắt.

Logic kiểm tra mẫu cần inference:

```python
def need_infer(item):
    return (
        "text" not in item
        or "timestamps" not in item
        or item.get("text") in [None, ""]
        or item.get("timestamps") in [None, []]
    )
```

Cách thiết kế này phù hợp với labeling dataset lớn vì notebook có thể chạy nhiều lần mà không làm lại các audio đã xử lý thành công.

---

## 7. FastAPI labeling service

Để phục vụ quá trình labeling, notebook dựng một service local bằng FastAPI. Service này giúp tách phần inference thành API rõ ràng, dễ gọi từ script labeling hoặc từ worker khác.

### 7.1. Endpoint `/health`

Endpoint `/health` dùng để kiểm tra server và trạng thái model.

Kết quả test:

```json
{
  "status": "ok",
  "model": "Qwen3-ASR",
  "device": "cuda:0",
  "language_default": "Vietnamese"
}
```

Kết quả này xác nhận model đã load thành công và đang chạy trên GPU.

### 7.2. Endpoint `/transcribe`

Endpoint `/transcribe` nhận một file audio và trả về transcript cùng word-level metadata.

Ví dụ format output:

```json
{
  "text": "...",
  "timestamps": [
    {
      "word": "hydroquinone",
      "confidence": 0.8287627696990967,
      "start": 2.64,
      "end": 3.20
    }
  ]
}
```

Output này phù hợp với mục tiêu của bài toán vì vừa tạo được transcript để fine-tune, vừa có thông tin confidence và timestamp để đánh giá chất lượng nhãn.

---

## 8. Cách tính confidence theo word

Notebook đã implement phần tính confidence từ logits/scores của mô hình. Đây là nền tảng để chấm điểm chất lượng transcript.

### 8.1. Confidence ở cấp token

Tại mỗi bước decode, mô hình tạo ra một vector logits. Xác suất của token được chọn được tính như sau:

$$
p_t = softmax(logits_t)[token_t]
$$

Trong đó:

- `logits_t`: vector logits tại bước sinh token thứ `t`.
- `token_t`: token được mô hình chọn tại bước `t`.
- `p_t`: xác suất mô hình gán cho token đã chọn.

Token có xác suất càng cao thì mô hình càng chắc chắn với lựa chọn đó.

### 8.2. Gom token thành word

Vì một word có thể gồm nhiều sub-token, notebook gom token thành word dựa trên các dấu hiệu của tokenizer, ví dụ:

- token bắt đầu bằng khoảng trắng,
- marker SentencePiece `▁`,
- marker BPE `Ġ`.

Sau khi gom token, confidence của một word được tính bằng xác suất thấp nhất trong các token tạo nên word đó:

$$
Conf(w_i) = \min_{t \in w_i} p_t
$$

Cách dùng `min` giúp score nghiêm ngặt hơn. Nếu một word có nhiều sub-token nhưng chỉ một sub-token có confidence thấp, toàn bộ word sẽ được đánh dấu là cần chú ý.

---

## 9. Phương pháp chấm điểm chất lượng audio-transcript

Do dataset chưa có ground truth transcript thủ công, tuần 3 sử dụng score nội bộ dựa trên word-level confidence. Score này chưa thay thế hoàn toàn WER/CER, nhưng đủ hữu ích để lọc dữ liệu pseudo label thành các nhóm chất lượng khác nhau.

Gọi:

- $c_{ij}$: confidence của word thứ $j$ trong audio $i$.
- $W_i$: số word trong audio $i$.
- $\tau$: ngưỡng confidence thấp, hiện dùng $\tau = 0.70$.
- $TimestampPresence_i$: tỷ lệ word có đủ trường `start` và `end`.

Mean confidence của audio:

$$
MeanConf_i = \frac{1}{W_i}\sum_{j=1}^{W_i} c_{ij}
$$

Tỷ lệ word rủi ro thấp chất lượng:

$$
LowRate_i = \frac{|\{c_{ij} < \tau \ \text{hoặc}\ c_{ij}=null\}|}{W_i}
$$

Score cho một audio:

$$
Score_i = 10 \times MeanConf_i \times TimestampPresence_i \times (1 - LowRate_i)
$$

Score được giới hạn trong khoảng 0 đến 10:

$$
Score_i = \min(10, \max(0, Score_i))
$$

Score cho toàn bộ phần dữ liệu đã label được tính theo trung bình có trọng số theo số word:

$$
DatasetScore = \frac{\sum_i W_i \times Score_i}{\sum_i W_i}
$$

Để tích hợp multi-model agreement, score sẽ được mở rộng thêm thành phần agreement giữa Qwen3-ASR và FunASR:

$$
Agreement_i = 1 - WER(Transcript^{Qwen}_i, Transcript^{FunASR}_i)
$$

Trong trường hợp không tách word chuẩn để tính WER, có thể dùng CER hoặc text similarity thay thế. Score mở rộng:

$$
FinalScore_i = 10 	imes MeanConf_i 	imes TimestampPresence_i 	imes Agreement_i 	imes (1 - LowRate_i)
$$

Với cách này, một audio chỉ được chấm cao khi transcript vừa có confidence tốt, vừa có timestamp đầy đủ, vừa có độ đồng thuận cao giữa hai mô hình ASR. Đây là cách chấm nghiêm ngặt hơn và phù hợp hơn cho mục tiêu chọn dữ liệu đưa vào fine-tune.

Cách tính này có ba ưu điểm:

1. Ưu tiên audio có confidence trung bình cao.
2. Phạt các audio có nhiều word confidence thấp hoặc thiếu confidence.
3. Trọng số theo số word giúp audio quá ngắn không làm lệch score của toàn dataset.
4. Thành phần agreement giúp phát hiện các trường hợp một model có thể tự tin nhưng transcript vẫn có rủi ro sai.

Lưu ý: timestamp có hai khía cạnh khác nhau. Trong kết quả tuần 3, `TimestampPresence` đạt 100% vì các word đều có trường `start` và `end`. Tuy nhiên, một phần timestamp có `start >= end`, nên độ mượt của alignment vẫn cần được kiểm tra riêng như một chỉ số phụ, không nên hiểu là toàn bộ timestamp đã hoàn hảo.

---

## 10. Kết quả thử nghiệm trên audio mẫu

Notebook đã thử nghiệm với file:

```text
/content/VietMed_un_513_s37N9G.wav
```

Transcript mẫu có nội dung liên quan đến tư vấn da liễu và xuất hiện nhiều thuật ngữ như `hydroquinone`, `abutin`, `serum c`, `phác đồ điều trị`.

Một số word confidence quan sát được:

| Word | Confidence |
|---|---:|
| thì | 0.9982 |
| có | 1.0000 |
| hydroquinone | 0.8288 |
| abutin | 0.5915 |
| serum | 0.8859 |
| c | 0.6409 |
| phác | 0.4983 |

Nhận xét:

- Các từ tiếng Việt phổ biến như “thì”, “có”, “là”, “chúng ta” thường có confidence rất cao.
- Các thuật ngữ y tế, tên hoạt chất hoặc từ ngoại lai như `hydroquinone`, `abutin`, `serum c` có confidence thấp hơn.
- Đây là đặc điểm phù hợp với dữ liệu VietMed, vì audio có nhiều tên thuốc, hoạt chất và từ vay mượn tiếng Anh.
- Word-level confidence giúp phát hiện chính xác các vị trí nên review, thay vì phải nghe lại toàn bộ audio.

---

## 11. Kết quả đánh giá trên split `000_050`

Phần đánh giá sơ bộ được áp dụng trên file label `VietMed_unlabeled_000_050.json`.

| Chỉ số | Giá trị |
|---|---:|
| Tổng số audio trong JSON | 7,975 |
| Audio đã có `text` và `timestamps` | 475 |
| Audio chưa có label | 7,500 |
| Tiến độ labeling | 5.96% |
| Tổng số word/timestamp đã sinh | 35,546 |
| Word có confidence hợp lệ | 35,305 |
| Word thiếu confidence | 241 |
| Mean confidence | 0.9755 |
| Median confidence | 1.0000 |
| Word confidence `< 0.70` | 1,195 |
| Word confidence `< 0.70` tính cả null | 1,436 |
| Tỷ lệ low-confidence tính cả null | 4.04% |
| Dataset score sơ bộ | **9.37/10** |

Kết quả cho thấy phần lớn pseudo label hiện có confidence cao. Mean confidence đạt 0.9755, median đạt 1.0000 và dataset score sơ bộ đạt 9.37/10 trên phần đã label. Điều này cho thấy pipeline có khả năng tạo nhãn tự động chất lượng tốt cho bước fine-tune ban đầu.

Tuy nhiên, score này cần được hiểu đúng phạm vi: đây là score dựa trên confidence nội bộ của mô hình và mới tính trên 475 audio đã label, chưa đại diện cho toàn bộ tập VietMed.

---

## 12. Phân nhóm chất lượng pseudo label

Dựa trên audio-level score, phần dữ liệu đã label được chia thành ba nhóm:

| Nhóm score | Số audio | Số word | Cách sử dụng đề xuất |
|---|---:|---:|---|
| `>= 8.5` | 452 | 34,721 | Ưu tiên đưa vào tập fine-tune ban đầu |
| `7.0 - 8.5` | 14 | 765 | Review các word confidence thấp trước khi dùng |
| `< 7.0` | 9 | 60 | Chạy lại bằng model khác hoặc kiểm tra thủ công |

Cách chia nhóm này giúp giảm manual labeling theo hướng có chọn lọc:

- Không cần nghe lại toàn bộ 475 audio đã label.
- Chỉ tập trung review các word hoặc audio có score thấp.
- Nhóm `>= 8.5` có thể được ưu tiên dùng trước để tạo tập fine-tune chất lượng cao.

---

## 13. Kiểm soát chất lượng timestamp và alignment

Trong kết quả hiện tại, tất cả word đã sinh đều có trường `start` và `end`, nên timestamp presence đạt 100%. Tuy nhiên, khoảng 22.00% word có `start >= end`, nghĩa là timestamp bị trùng hoặc duration bằng 0.

Điều này không nhất thiết làm transcript sai, vì bài toán fine-tune ASR chủ yếu cần cặp audio-transcript. Tuy nhiên, nếu dùng timestamp cho review thủ công, cắt đoạn audio hoặc huấn luyện các pipeline cần alignment chính xác, cần bổ sung bước hậu xử lý alignment.

Hướng xử lý đề xuất:

1. Tách riêng hai chỉ số: `TimestampPresenceRate` và `TimestampDurationValidRate`.
2. Giữ transcript nếu confidence cao nhưng đánh dấu các word có `start >= end` để review khi cần timestamp.
3. Với các audio có nhiều timestamp lỗi, chạy lại forced aligner hoặc dùng aligner khác để cải thiện alignment.

---

## 14. Tiến độ tuần 3

### 14.1. Các phần đã hoàn thành

- Load thành công **Qwen3-ASR-1.7B** trên Colab GPU.
- Bật **Qwen3-ForcedAligner-0.6B** để lấy timestamp theo word.
- Xây dựng hàm transcribe một file audio.
- Xây dựng hàm tính token confidence từ logits/scores.
- Gom token thành word và tính word confidence.
- Xây dựng FastAPI local gồm `/health` và `/transcribe`.
- Test thành công API `/transcribe` trên audio mẫu.
- Output API đã có đủ `text`, `word`, `confidence`, `start`, `end`.
- Chạy pseudo labeling và ghi kết quả cho **475/7,975 audio** trong split `000_050`.
- Tính được score chất lượng sơ bộ cho phần dữ liệu đã label dựa trên word-level confidence.

### 14.2. Các phần đang tiếp tục

- Chạy tiếp labeling cho **7,500 audio còn lại** trong split `000_050`.
- Mở rộng pipeline sang các split tiếp theo sau khi checkpoint ổn định.
- Xuất file `dataset_score.csv` để so sánh chất lượng giữa các split.
- Tạo file review riêng chứa audio/word confidence thấp.
- Tích hợp **FunASR** làm model thứ hai để tính agreement với Qwen3-ASR.
- Kiểm tra các word có `confidence = null` hoặc timestamp `start >= end`.

---

## 15. Chiến lược giảm manual labeling

Pipeline hiện tại cho phép chuyển từ manual labeling toàn bộ sang manual review có chọn lọc.

Quy tắc sử dụng pseudo label:

| Điều kiện | Quyết định |
|---|---|
| Audio score `>= 8.5` | Dùng trực tiếp cho fine-tune ban đầu |
| Audio score `7.0 - 8.5` | Chỉ review các word confidence thấp |
| Audio score `< 7.0` | Chạy lại bằng model khác hoặc kiểm tra thủ công |
| Word confidence `< 0.70` | Đưa vào danh sách cần review |
| Word confidence `null` | Đánh dấu rủi ro, không tính là chắc chắn |
| Timestamp `start >= end` | Đánh dấu lỗi alignment, ưu tiên sửa nếu cần timestamp |

Danh sách word cần review có thể được tạo tự động:

```python
low_conf_words = [
    word for word in timestamps
    if word["confidence"] is None or word["confidence"] < 0.70
]
```

Nhờ đó, người gán nhãn chỉ cần nghe lại các đoạn có rủi ro cao, thay vì nghe lại toàn bộ audio.

---

## 16. Kế hoạch tuần tiếp theo

Trong tuần tiếp theo, pipeline sẽ được mở rộng theo hai hướng: hoàn thành labeling và tăng độ tin cậy của pseudo label bằng model thứ hai.

Các việc cần thực hiện:

1. Hoàn thiện cơ chế checkpoint để tránh lỗi khi ghi file JSON trên Google Drive.
2. Chạy tiếp pseudo labeling cho **7,500 audio còn lại** trong split `000_050`.
3. Sau khi hoàn thành split `000_050`, mở rộng sang các split tiếp theo trong folder `unlabeled_medical_data`.
4. Tự động xuất báo cáo chất lượng cho từng split, gồm số audio đã label, tổng số word, mean confidence, tỷ lệ confidence thấp, tỷ lệ confidence null và dataset score.
5. Sử dụng thêm **FunASR** như một mô hình ASR thứ hai để:
   - chạy lại các audio có score thấp,
   - so sánh agreement giữa Qwen3-ASR và FunASR,
   - chọn transcript đáng tin cậy hơn cho các audio khó,
   - cải thiện chất lượng pseudo label trước khi fine-tune.
6. Chuẩn bị tập fine-tune ban đầu bằng cách ưu tiên nhóm audio score `>= 8.5`.

Sau khi có đủ pseudo label chất lượng cao, mô hình ASR fine-tuned có thể được dùng để relabel lại các mẫu còn yếu, tạo vòng lặp cải thiện dữ liệu theo hướng semi-supervised learning.

---

## 17. Kết luận

Trong tuần 3, pipeline pseudo labeling cho VietMed ASR đã chuyển từ mức thử nghiệm trên audio đơn lẻ sang mức chạy thực tế trên một phần split `000_050`. Mô hình **Qwen3-ASR-1.7B** kết hợp **Qwen3-ForcedAligner-0.6B** đã tạo được transcript, timestamp và word-level confidence cho **475 audio** đầu tiên.

Phần dữ liệu đã label có **35,546 word/timestamp**, mean confidence đạt **0.9755**, tỷ lệ word rủi ro thấp hơn `0.70` tính cả null là **4.04%**, và dataset score sơ bộ đạt **9.37/10**. Kết quả này cho thấy pseudo labeling bằng ASR mạnh kết hợp confidence score là một hướng khả thi để giảm manual labeling cho VietMed.

Điểm quan trọng nhất của tuần 3 là pipeline không chỉ sinh transcript, mà còn xây dựng được cơ chế định lượng chất lượng nhãn. Ngoài score dựa trên word-level confidence, pipeline cũng đang được mở rộng bằng multi-model agreement với FunASR để kiểm tra mức độ đồng thuận giữa hai mô hình ASR. Nhờ đó, dữ liệu pseudo label có thể được lọc thành các nhóm rõ ràng: dùng trực tiếp, review chọn lọc hoặc chạy lại bằng model khác. Đây là nền tảng cần thiết để tạo tập dữ liệu chất lượng cao phục vụ fine-tune ASR ở các bước tiếp theo.

---

## 18. Tài liệu đính kèm

- **GitHub Repository:** https://github.com/phuvo05/Fun-ASR-VietMedLabeling
- **Google Colab notebook:** https://colab.research.google.com/drive/1izdKoJ0VQJE4DVTvlVE0nkEKEUvy672e?usp=sharing
- **Folder dữ liệu cần label:** https://drive.google.com/drive/folders/1Wrofm2FkjYngpxR1ysmkm0XvFiGEvyFM
