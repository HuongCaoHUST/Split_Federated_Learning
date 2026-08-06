# Phân tích toán học Dynamic Cut trong Split Federated Learning

> Cập nhật: 2026-08-05  
> Phạm vi: YOLO11, các cut `5`, `10`, `15`, `20`, nhiều edge client và **một shared dynamic server model**.

## 1. Mục đích

Tài liệu này lưu lại mô hình toán, kết quả audit code và protocol thực nghiệm để so sánh:

- Một cut chung cho mọi edge client (`uniform cut`).
- Mỗi edge client có cut riêng (`dynamic/heterogeneous cut`).

Các câu hỏi trọng tâm:

1. Dynamic cut có lợi gì về latency, RAM và communication?
2. Dynamic cut chỉ thay đổi nơi tính toán hay còn thay đổi thuật toán tối ưu?
3. Code hiện tại là virtual full-model aggregation hay shared sequential server update?
4. Đánh giá dynamic cut so với uniform cut thế nào cho công bằng?

## 2. Ký hiệu

Có `K` edge client và một training server. Canonical full model là:

```math
W=\{W_1,W_2,\ldots,W_L\}.
```

Client `k` có cut `\sigma_k`:

- Layer `1..\sigma_k` chạy trên edge `k`.
- Layer `\sigma_k+1..L` chạy trên server.
- Dữ liệu vẫn forward và backward qua toàn mạng.

Với layer `l`:

```math
\mathcal E_l=\{k\mid l\leq\sigma_k\},
\qquad
\mathcal S_l=\{k\mid l>\sigma_k\}.
```

Với trọng số dữ liệu `p_k=n_k/\sum_j n_j`, định nghĩa server-side data mass:

```math
q_l(\boldsymbol\sigma)=\sum_{k:\sigma_k<l}p_k.
```

`q_l` là tỷ lệ dữ liệu chạy layer `l` trên shared server.

## 3. YOLO11 skip connection và boundary activation

Server không cần sở hữu lại module nông hơn cut; edge phải gửi mọi activation đi qua biên cut do skip connection.

| Cut | Activation edge gửi cho server |
|---:|---|
| 5 | output layer `4`, `5` |
| 10 | output layer `4`, `6`, `10` |
| 15 | output layer `15`, `13`, `10` |
| 20 | output layer `20`, `10`, `16`, `19` |

Nguồn: [`model/YOLO11n_custom.py`](model/YOLO11n_custom.py), phần `YOLO11_DYNAMIC_SERVER` và các lớp `YOLO11_EDGE_*`.

Kết quả kiểm tra numerical:

- Full YOLO11 và split route ở cả bốn cut có `max_abs_diff=0.0` khi cùng weights.
- Một shared server với `min_cut=5` route đồng thời cả bốn cut vẫn khớp full model.
- Tất cả boundary activation nhận gradient.
- Server layer đã chạy trên edge được skip và không nhận gradient ở route tương ứng.

## 4. Runtime model không phải năm pipeline stage rời nhau

Với bốn client ở cut `5`, `10`, `15`, `20`:

| Thành phần | Layer sở hữu |
|---|---|
| Edge cut 5 | `0..5` |
| Edge cut 10 | `0..10` |
| Edge cut 15 | `0..15` |
| Edge cut 20 | `0..20` |
| Dynamic server | `6..23`, vì `min_cut=5` |

Các edge prefix chồng lấn. Có thể chia canonical model thành các vùng aggregation:

```text
P1: 0..5
P2: 6..10
P3: 11..15
P4: 16..20
P5: 21..23
```

## 5. Kết quả audit thuật toán hiện tại

### 5.1 Loại thuật toán

Code hiện tại là **shared sequential server update**, không phải **virtual full-model aggregation per client**.

Server chỉ tạo một `self.model` và một optimizer. Với từng intermediate payload, [`src/train.py`](src/train.py) thực hiện ngay:

```python
outputs = self.model(client_tensors, cut_layer=cut_layer)
loss.backward()
self.optimizer.step()
```

Luồng thực tế:

```text
batch client 1 -> update shared server model
batch client 2 -> dùng model đã bị client 1 thay đổi
batch client 3 -> dùng model đã bị client 1 và 2 thay đổi
...
```

Không có `server_models[client_id]` hay `server_optimizers[client_id]`; không thể khôi phục suffix update độc lập `Delta W_(k,l)` cho mỗi client.

### 5.2 Client nào ảnh hưởng layer nào?

- Với `k` thuộc `E_l`, client `k` update edge copy riêng `W_(k,l)^edge`.
- Với `k` thuộc `S_l`, dữ liệu client `k` update cùng shared server copy `W_l^srv` theo thứ tự queue.
- Mỗi batch vẫn chạy mỗi layer đúng một lần; không có double forward/backward.
- Server-side attribution bị mất; `route_batch_counts` gom theo cut, không theo client.

Ví dụ `K=8`, hai client ở mỗi cut:

```text
[5,5,10,10,15,15,20,20]
```

| Layer | Independent edge copies | Shared server copy |
|---|---:|---:|
| `0..5` | 8 client | Không |
| `6..10` | 6 client | Trộn tuần tự 2 client cut 5 |
| `11..15` | 4 client | Trộn tuần tự 4 client cut 5/10 |
| `16..20` | 2 client | Trộn tuần tự 6 client cut 5/10/15 |
| `21..23` | Không | Trộn tuần tự cả 8 client |

## 6. Aggregation thực tế

Với một server worker, code gần với:

```math
W_l^{merge}
=
\frac{
\sum_{k\in\mathcal E_l}B_kW_{k,l}^{edge}
+B_{\mathcal S_l}W_l^{srv}
}{\sum_kB_k},
\qquad
B_{\mathcal S_l}=\sum_{k\in\mathcal S_l}B_k.
```

`B_k` hiện là số batch, không phải số sample.

Nguồn code:

- Edge gửi `len(train_loader)` làm `nb_train`: [`src/train.py`](src/train.py).
- Server tăng route count mỗi intermediate payload: [`src/train.py`](src/train.py).
- Edge/server candidates được gom layer-wise: [`src/server.py`](src/server.py), hàm `merged_model`.

Coverage check bảo đảm tổng exposure của mỗi parameterized layer bằng tổng edge batch. Không thấy lỗi thiếu source layer hoặc cộng cùng execution path hai lần.

Vấn đề là:

```math
W_l^{srv}
\neq
\frac{\sum_{k\in\mathcal S_l}B_kW_{k,l}^{server}}{B_{\mathcal S_l}},
```

vì `W_l^srv` là kết quả của một optimizer chạy tuần tự trên cả nhóm `S_l`, không phải trung bình local server models độc lập.

## 7. First-order analysis

Giả sử minh họa:

- `p_k=1/K`.
- Mỗi client đóng góp một gradient `g_(k,l)`.
- Bỏ qua momentum, Hessian và higher-order terms.

Gọi `s_l=|S_l|`. Shared server update gần đúng:

```math
W_l^{srv}\approx W_l^t-\eta\sum_{k\in\mathcal S_l}g_{k,l}.
```

Sau merge hiện tại:

```math
\Delta W_l^{dyn}
\approx
-\frac{\eta}{K}
\left[
\sum_{k\in\mathcal E_l}g_{k,l}
+s_l\sum_{k\in\mathcal S_l}g_{k,l}
\right].
```

Sample-weighted global gradient lý tưởng:

```math
\Delta W_l^{ideal}
=
-\frac{\eta}{K}
\left[
\sum_{k\in\mathcal E_l}g_{k,l}
+\sum_{k\in\mathcal S_l}g_{k,l}
\right].
```

Sai lệch first-order trong mô hình đơn giản:

```math
B_l(\boldsymbol\sigma)
\approx
-\frac{\eta(s_l-1)}{K}
\sum_{k\in\mathcal S_l}g_{k,l}.
```

Huấn luyện thật còn phụ thuộc gradient tại các server weight khác nhau, momentum/Adam state, BatchNorm, queue arrival order và số local batch/epoch.

## 8. Uniform cut so với dynamic cut

### 8.1 Uniform cut

Nếu mọi client dùng cut `c`:

```math
q_l=
\begin{cases}
0,&l\leq c,\\
1,&l>c.
\end{cases}
```

Ví dụ uniform cut 10:

| Layer | Cách huấn luyện |
|---|---|
| `0..10` | `K` independent edge models rồi FedAvg |
| `11..23` | Một shared server model update tuần tự trên `K` client |

Đây gần với SFL-V2 có một cut chung.

### 8.2 Dynamic cut

Với `[5,5,10,10,15,15,20,20]` và số mẫu bằng nhau:

| Layer | `q_l` | Diễn giải |
|---|---:|---|
| `0..5` | 0 | Federated edge hoàn toàn |
| `6..10` | 0.25 | 75% edge, 25% shared server |
| `11..15` | 0.50 | 50% edge, 50% shared server |
| `16..20` | 0.75 | 25% edge, 75% shared server |
| `21..23` | 1.00 | Shared server hoàn toàn |

Uniform cut tạo bước nhảy `0 -> 1`; dynamic tạo staircase `0 -> 0.25 -> 0.5 -> 0.75 -> 1`.

Dynamic hiện tại là hybrid layer-wise giữa FL-style local training và centralized sequential training. Không có định lý tổng quát rằng hybrid này luôn chính xác hơn uniform cut.

## 9. Baseline hướng B: canonical-gradient split

Tệp `config_canonical_cut5.yaml` triển khai baseline kiểm chứng cho **một edge,
một server, uniform cut `5`**. Server giữ một `YOLO11_Full` canonical và một
optimizer duy nhất. Với mỗi batch:

```text
edge forward (layers 0..5)
  -> server forward/backward (layers 6..23)
  -> edge backward (layers 0..5)
  -> server cài gradient prefix + BN buffers
  -> server optimizer.step() đúng một lần
```

Vì tất cả gradient được đánh giá tại cùng `W^t` và optimizer chỉ step một lần,
update bằng full-model training cho cùng batch (trong sai số số học). Script
`verify_canonical_cut5.py` kiểm tra loss, mọi gradient và state sau một SGD step
trên synthetic YOLO batch; kết quả mong đợi là sai khác bằng 0.

Baseline này chưa mở rộng sang nhiều edge hay nhiều cut. Với nhiều edge, server
phải đợi gradient prefix/suffix của toàn bộ client được tính từ cùng snapshot,
weight theo số sample, aggregate từng canonical layer rồi mới step một lần. Đó
là phần cần thiết để chứng minh cut-invariance cho dynamic cut, không phải chỉ
nhân bản protocol hiện tại theo số client.

## 9. Bảo đảm toán học về system cost

Gọi `C={5,10,15,20}`. Uniform search space là:

```math
\mathcal C_{uniform}
=\{[5,\ldots,5],[10,\ldots,10],[15,\ldots,15],[20,\ldots,20]\}.
```

Dynamic search space:

```math
\mathcal C_{dynamic}=\mathcal C^K.
```

Do `C_uniform` là tập con của `C_dynamic`, với cùng cost model và không ép buộc dùng đủ mọi cut:

```math
\boxed{
\min_{\boldsymbol\sigma\in\mathcal C_{dynamic}}T(\boldsymbol\sigma)
\leq
\min_{\boldsymbol\sigma\in\mathcal C_{uniform}}T(\boldsymbol\sigma)
}.
```

Đây là bảo đảm về nghiệm tối ưu hệ thống, không phải accuracy. Bất đẳng thức thường nghiêm khi thiết bị khác compute, RAM hoặc network.

## 10. Latency model cho một shared server

Đặt:

- `b_k`: số batch client `k`.
- `C^e_(k,c)`: edge forward + backward tại cut `c`.
- `C^s_c`: server forward + backward cho route cut `c`.
- `A_c`: boundary activation/gradient bytes.
- `R_k^up`, `R_k^down`: uplink/downlink.

```math
D_{k,c}
=b_k\left(
C^e_{k,c}
+\frac{A_c}{R_k^{up}}
+\frac{A_c}{R_k^{down}}
\right),
```

```math
S(\boldsymbol\sigma)=\sum_{k=1}^{K}b_kC^s_{\sigma_k}.
```

Lower bound đơn giản:

```math
T_{round}
\gtrsim
\max\left\{\max_kD_{k,\sigma_k},S(\boldsymbol\sigma)\right\}.
```

Nên dùng discrete-event queue simulation vì mỗi edge gửi batch rồi chờ gradient.

Nếu có `n_5,n_10,n_15,n_20` client và số batch bằng nhau:

```math
S\propto
n_5C_{6:10}
+(n_5+n_{10})C_{11:15}
+(n_5+n_{10}+n_{15})C_{16:20}
+KC_{21:23}.
```

## 11. Communication profile của YOLO11

Đo tại input `640x640`, batch size 1, FP32:

| Cut | Boundary tensors | Forward | Activation + gradient | Edge parameters |
|---:|---|---:|---:|---:|
| 5 | L4, L5 | 4.096 MB | 8.192 MB/ảnh | 0.890 MB |
| 10 | L4, L6, L10 | 4.506 MB | 9.011 MB/ảnh | 5.462 MB |
| 15 | L15, L13, L10 | 7.782 MB | 15.565 MB/ảnh | 5.907 MB |
| 20 | L20, L10, L16, L19 | 3.072 MB | 6.144 MB/ảnh | 7.121 MB |

Chưa gồm labels, pickle và protocol overhead. Do skip connection, communication không đơn điệu: cut 15 truyền nhiều nhất; cut 20 truyền ít nhất nhưng edge compute/RAM lớn nhất.

## 12. Metadata và model version

Intermediate payload hiện có `client_id`, `cut_layer`, `epoch`. Model checkpoint update có `client_id`, `layer_id`, `epoch`.

Thiếu:

```text
stable_device_id
round_id
model_version
global_snapshot_hash
batch_id
num_samples
activation_layer_ids
```

Server có chặn stale epoch và giữ future-epoch payload, nhưng các batch cùng epoch vẫn được tính trên những phiên bản shared server weight khác nhau.

Coordinator merge/validation mỗi epoch nhưng chỉ broadcast global model sau toàn bộ `num_epochs`. Nếu `round` là outer round gồm `num_epochs`, các thành phần bắt đầu outer round từ cùng snapshot; nếu coi mỗi epoch là global round thì epoch sau không bắt đầu từ model vừa merge.

## 13. Full-model merge và validation

Canonical mapping hiện hoạt động:

- Edge checkpoint map với offset 0.
- Dynamic server checkpoint map với offset `min_cut+1`.
- Merge tạo `YOLO11_Full` và kiểm tra exposure.
- Validation loss và NMS chạy được.

Đã kiểm tra bốn cut cùng lúc: đủ source state; merged model khớp full YOLO11 với `max_abs_diff=0.0` khi source weights giống nhau; validation loss hữu hạn và NMS thành công.

Checkpoint chứa custom `YOLO11_Full`. Nếu cần strict-load state dict vào Ultralytics `DetectionModel` nguyên bản, cần map `layers.N.* -> model.N.*`.

## 14. Protocol so sánh dynamic và uniform

Uniform baselines bắt buộc với `K=8`:

```text
U5  = [5,5,5,5,5,5,5,5]
U10 = [10,10,10,10,10,10,10,10]
U15 = [15,15,15,15,15,15,15,15]
U20 = [20,20,20,20,20,20,20,20]
```

Dynamic phải so với **best uniform**, không chỉ một cut tùy ý. Dynamic candidates tối thiểu:

```text
D_balanced = [5,5,10,10,15,15,20,20]
D_latency  = vector tối ưu predicted latency
D_comm     = vector tối ưu communication
D_resource = vector theo compute/RAM/network từng thiết bị
```

Metrics:

```math
G_T=\frac{T_{best\ uniform}}{T_{dynamic}},
\qquad
G_C=\frac{C_{best\ uniform}}{C_{dynamic}}.
```

```math
\Delta A_{update}=A_{dynamic}(N)-A_{uniform}(N),
```

```math
\Delta A_{time}=A_{dynamic}(T)-A_{uniform}(T),
```

```math
G_{TTA}
=\frac{T_{uniform}(A_{target})}{T_{dynamic}(A_{target})}.
```

Global update deviation từ cùng snapshot:

```math
D_W
=
\frac{\|W_{dynamic}^{t+1}-W_{uniform}^{t+1}\|_2}{\|W^t\|_2}.
```

Nên báo cáo `D_W` riêng cho `0..5`, `6..10`, `11..15`, `16..20`, `21..23`.

Experimental controls:

- Cùng initialization, data partition, batch order, sample count.
- Cùng seed, learning rate và optimizer.
- Cùng client scheduling order khi làm ablation.
- Ít nhất 3 đến 5 seed, báo cáo mean và confidence interval.
- Chạy cả deterministic round-robin scheduling và natural RabbitMQ arrival order.

## 15. Tìm cut vector tốt nhất

Với `K=8` và bốn cut có `4^8=65,536` labeled assignments; đủ nhỏ để exhaustive-score bằng profiler và queue simulator.

Nếu thiết bị hoàn toàn đồng nhất và chỉ histogram quan trọng thì chỉ có:

```math
\binom{8+4-1}{4-1}=165
```

phân bố.

Quy trình:

1. Profile đủ `8 x 4 = 32` cặp device-cut.
2. Đo edge/server compute, boundary bytes, RAM và network.
3. Loại assignment vi phạm RAM/SLA.
4. Mô phỏng single-server queue cho toàn bộ assignments.
5. Giữ 10 đến 20 điểm trên Pareto frontier.
6. Pilot train từ cùng snapshot và seed.
7. Chọn theo `time-to-target mAP` hoặc `mAP50-95 / wall-clock`.

Logger hiện có latency/payload nhưng thiếu `cut_layer`, `num_samples`, stable device ID và queue wait. `client_id` được sinh ngẫu nhiên khi process khởi động, chưa phù hợp profile lâu dài theo thiết bị vật lý.

## 16. Giữ một server model nhưng giảm order dependence

Cải tiến tối thiểu là synchronized gradient accumulation:

```python
server_optimizer.zero_grad()

for client_batch in synchronization_window:
    loss_k = forward(client_batch)
    weighted_loss_k.backward()

server_optimizer.step()
```

Mục tiêu: không update parameter giữa các client trong cùng cửa sổ, gradient dùng cùng server model version, giảm phụ thuộc RabbitMQ order và vẫn chỉ giữ một shared server model.

Cần barrier/window, `batch_id`, `model_version`, sample weighting và giữ boundary gradients tới khi cửa sổ hoàn thành. Cách này gần MiniBatch-SFL hơn nhưng chưa tự động làm heterogeneous-cut algorithm cut-invariant vì edge layers vẫn dùng independent optimizers.

Muốn canonical update sạch hơn mà vẫn một server model:

1. Server giữ một full canonical state.
2. Edge gửi prefix gradient/delta kèm client/layer/sample/version metadata.
3. Server tích lũy server-side gradient.
4. Mỗi global layer được aggregate từ mọi client đúng một lần.
5. Global optimizer update canonical layer đúng một lần.

## 17. Kết luận cần ghi nhớ

1. Skip connection và canonical mapping đúng cho cut `5`, `10`, `15`, `20`.
2. Merge tạo được full model và validation hoạt động.
3. Code là shared sequential server update, không phải virtual full-model FedAvg.
4. Không double forward/backward, nhưng không có suffix update độc lập theo client.
5. Aggregation dùng số batch, không dùng số sample thực tế.
6. Dynamic cut có lợi thế toán học về best achievable system cost vì uniform là tập con của dynamic.
7. Dynamic cut không được bảo đảm accuracy tốt hơn uniform với SFL-V2/shared sequential server.
8. Dynamic cut hiện thay đổi `q_l`, optimizer trajectory, arrival order và effective layer-wise update; không chỉ thay đổi nơi tính toán.
9. So sánh đúng phải dùng best uniform và báo cáo cả fixed-update lẫn fixed-wall-clock.
10. Metric triển khai quan trọng nhất là `time-to-target mAP`, cùng layer-wise update deviation.

## 18. Tài liệu tham khảo

- [SplitFed: When Federated Learning Meets Split Learning](https://arxiv.org/abs/2004.12088)
- [When MiniBatch SGD Meets SplitFed Learning](https://arxiv.org/abs/2308.11953)
- [The Impact of Cut Layer Selection in Split Federated Learning](https://arxiv.org/abs/2412.15536)
- [Efficient Parallel Split Learning over Resource-constrained Wireless Edge Networks](https://arxiv.org/abs/2303.15991)
- [Split Federated Learning Over Heterogeneous Edge Devices](https://arxiv.org/abs/2411.13907)

## 19. Checklist cho agent tiếp theo

- [ ] `K`, cut vector và mapping ổn định `device_id -> cut`.
- [ ] Mọi edge/server bắt đầu outer round từ cùng `model_version`.
- [ ] Payload có client, round, version, batch, sample metadata.
- [ ] Boundary activation IDs đúng YOLO graph.
- [ ] Server scheduling deterministic khi làm ablation.
- [ ] Weighting dùng sample count thay vì chỉ batch count.
- [ ] Không diễn giải shared sequential state như nhiều local updates độc lập.
- [ ] Đã chạy `U5`, `U10`, `U15`, `U20`.
- [ ] Dynamic được so với best uniform.
- [ ] Báo cáo latency, communication, RAM, mAP/update, mAP/time và time-to-target.
- [ ] Kiểm tra layer-wise `D_W` để phát hiện cut-dependent optimization drift.
