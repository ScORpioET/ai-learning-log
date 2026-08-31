# generate_captions.py v0.6 → v0.7 對比

20 張隨機抽樣（14 train + 6 val，`random.seed(42)`），`num_objects` 是該圖動態物件總數（occlusion 過濾後）。

| split | file | num_objects | v0.6 caption | v0.7 caption |
|---|---|---|---|---|
| train | video-BjSfmxLQqCGjg8tya-frame-000645-gyu9wMM3z5pKRFe2s.jpg | 4 | Nearby ahead there is a vehicle; nearby on the left there are two pedestrians. | Two pedestrians, the nearest on the left; a vehicle nearby ahead. |
| train | video-iTrM5ner4KQGLvpdC-frame-003652-p6NEbCDHXnaad3ywv.jpg | 2 | At medium distance ahead there is a pedestrian; at medium distance ahead there is a motorcycle. | A pedestrian ahead; a motorcycle ahead. |
| train | video-JFiGa9oAAN2queuLt-frame-007401-ZmvY558YSMBk9pH6n.jpg | 13 | Nearby on the right there are two pedestrians; at medium distance ahead there are several cars. | Many pedestrians, one nearby on the right. |
| train | video-vvYEi6EdCrYXeHwpD-frame-008561-HW94DXomd3bxFyzYG.jpg | 19 | Nearby on the right there are two cars; nearby on the left there is a car. | Many cars, one nearby on the right. |
| train | video-XNM8eHsvxCMJ6peuj-frame-001820-zFs26Bk6AiG7Yy6ZP.jpg | 11 | Nearby ahead there is a pedestrian; nearby ahead there is a motorcycle. | Several pedestrians, the nearest ahead. |
| train | video-r68Yr9RPWEp5fW2ZF-frame-017039-RZ4PxY4BsHjKtwvaj.jpg | 7 | Nearby ahead there is a car; at medium distance on the right there are several pedestrians. | Several cars, one nearby ahead; three pedestrians, the nearest on the right. |
| train | video-AEZAaF8epmvQv5Nfj-frame-008530-fESENfax6PWTrQdDh.jpg | 7 | Night: nearby on the right there are two cars; nearby ahead there is a bicycle. | Night: three bicycles, one nearby ahead; two cars, one nearby on the right. |
| train | video-wGdqQimawNX3RYv76-frame-000300-uRKkwqKJoFmPE7gRH.jpg | 9 | Nearby on the left there is a bus; nearby on the left there is a car. | Many cars, the nearest on the left. |
| train | video-rnZseY2bPiYhWEF2m-frame-009213-JvJsw2q39CkzLADd2.jpg | 17 | Nearby ahead there are two cars; at medium distance on the right there are two pedestrians. | Many pedestrians, one on the right. |
| train | video-TxNGEKRyWMf6Q7hS2-frame-000738-vYCtWP4iveZR69cyp.jpg | 10 | Nearby on the left there are two cars; nearby ahead there are two cars. | Many cars, the nearest on the left. |
| train | video-nodg7n2aFzCWFJBnv-frame-001510-jRgGKjKZyM9kToTQP.jpg | 24 | Nearby on the right there are several pedestrians; nearby on the right there is a car. | Many pedestrians, one nearby on the right. |
| train | video-AEZAaF8epmvQv5Nfj-frame-016391-NmZXAkAkFnwr6DmNR.jpg | 30 | Night: nearby on the left there are several cars; nearby on the right there are several cars. | Night: many cars, the nearest on the left. |
| train | video-zNFzcc9wW8XB4QwTa-frame-000258-yyWjd2KhKqQBsYDXr.jpg | 2 | Cloudy: nearby on the left there is a pedestrian; nearby ahead there is a pedestrian. | Cloudy: two pedestrians, one nearby on the left. |
| train | video-DJYDQTZH7GLCqw3Tx-frame-002717-FHqeyTMLK3g4kscGx.jpg | 35 | Cloudy: nearby on the right there are two cars; nearby on the left there are several cars. | Cloudy: many cars, the nearest on the right. |
| val | video-JhYLiFCieHQHaY8o7-frame-004855-x4qftk8GSFonh27r8.jpg | 8 | Night: nearby ahead there is a car; at medium distance on the left there is a bicycle. | Night: several pedestrians, one on the right. |
| val | video-JhYLiFCieHQHaY8o7-frame-005260-LpPvz3BwXrbzv7dYf.jpg | 2 | Night: nearby on the right there is a pedestrian; at medium distance ahead there is a pedestrian. | Night: two pedestrians, one nearby on the right. |
| val | video-mKfYgxHA8ZZmXvw56-frame-007670-5Ey9SjHq4Y6TvYbkz.jpg | 13 | Nearby ahead there is a pedestrian; nearby ahead there is a car. | Many cars, the nearest ahead. |
| val | video-Qk8msXvMopoYNDdco-frame-001810-bvkpamYzyxqX5Tmmm.jpg | 11 | Nearby on the left there is a car; at medium distance on the right there are several pedestrians. | Many pedestrians, the nearest on the right. |
| val | video-YQpCvGJxowy9uhkCw-frame-003600-HRaMd8WQk6AC9Q8cw.jpg | 18 | Nearby on the right there are two cars; nearby ahead there are two cars. | Many cars, one nearby on the right. |
| val | video-Qk8msXvMopoYNDdco-frame-004080-uGfErrjPcFthkPqXA.jpg | 12 | Nearby on the right there is a car; nearby on the left there are two cars. | Many cars, the nearest on the right. |

## 觀察

- 高 `num_objects` 的圖（13/17/19/24/30/35）v0.6 全部只講 2 個 group，且常常是「同類別重複」（例如 num_objects=9 那張，2 個 group 其實都是講車，一個講 bus 一個講 car，完全沒提到總數）；v0.7 一律先講類別總數（many/several），畫面上真的有很多車/行人時不會被截斷成只看到 2 台。
- `num_objects=2` 的兩張圖（mid distance）：v0.6 印出 "at medium distance ahead"，v0.7 省略距離詞直接變 "ahead"（Level 1 mid-omit 修正）。
