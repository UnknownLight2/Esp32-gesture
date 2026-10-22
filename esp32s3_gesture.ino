/*
  ESP32-S3 Gesture Recognition (fist / palm / peace / thumbs_up / none)

  - Mặc định phát WiFi AP "ESP32-GESTURE" (pass 12345678) tại 192.168.4.1
  - Mở http://192.168.4.1 để xem cam; in địa chỉ lên Serial Monitor (115200)
  - Trang web: video + kết quả nhận dạng + xác suất từng lớp
  - /status trả JSON;  stream MJPEG ở cổng 81 (/stream) -> dùng được với collect_data.py
  - Nếu CHƯA có model_data.h -> chạy chế độ chỉ stream cam (để tìm IP + thu dataset)

  Arduino IDE:
    Board: ESP32S3 Dev Module  (package "esp32" by Espressif)
    PSRAM: OPI PSRAM (N16R8) hoặc QSPI PSRAM (N8R2) - tùy module của bạn
    Partition Scheme: Huge APP (3MB No OTA)   |  USB CDC On Boot: Enabled
    Library: "Chirale_TensorFlowLite" (Library Manager)
*/

#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"

#if __has_include("model_data.h")
  #define HAS_MODEL 1
  #include <Chirale_TensorFlowLite.h>
  #include "tensorflow/lite/micro/micro_interpreter.h"
  #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
  #include "tensorflow/lite/schema/schema_generated.h"
  #include "model_data.h"
#else
  #define HAS_MODEL 0
#endif

// ================= CẤU HÌNH =================
const char* AP_SSID   = "ESP32-GESTURE";
const char* AP_PASS   = "12345678";

// Mapping tham khảo theo IoTLabs Bài 10. Một bài khác trong cùng series đưa
// mapping CAM phổ biến khác; xem README và xác nhận revision trước khi kết luận.
#define CAMERA_MODEL_IOTLABS_S3_CAM

const float CONF_THRESHOLD = 0.70f;   // dưới ngưỡng này -> coi là "không nhận dạng"
// ============================================

#if defined(CAMERA_MODEL_IOTLABS_S3_CAM)
  #define PWDN_GPIO_NUM -1
  #define RESET_GPIO_NUM -1
  #define XCLK_GPIO_NUM 15
  #define SIOD_GPIO_NUM 4
  #define SIOC_GPIO_NUM 5
  #define Y9_GPIO_NUM 16
  #define Y8_GPIO_NUM 17
  #define Y7_GPIO_NUM 18
  #define Y6_GPIO_NUM 8
  #define Y5_GPIO_NUM 3
  #define Y4_GPIO_NUM 46
  #define Y3_GPIO_NUM 9
  #define Y2_GPIO_NUM 10
  #define VSYNC_GPIO_NUM 6
  #define HREF_GPIO_NUM 7
  #define PCLK_GPIO_NUM 13
#else
  #error "Hãy chọn một CAMERA_MODEL_..."
#endif

static const int IMG = 96;
static const int MAX_CLASSES = 16;

// ---------- trạng thái dùng chung giữa các task ----------
static SemaphoreHandle_t gMutex;
static uint8_t* gJpg = nullptr;
static size_t gJpgLen = 0;
static uint32_t gFrameId = 0;
static float gProbs[MAX_CLASSES] = {0};
static int gBest = -1;
static float gConf = 0;
static uint32_t gInferMs = 0;

// ---------- TFLite Micro ----------
#if HAS_MODEL
constexpr int kArenaSize = 400 * 1024;      // đặt trong PSRAM
static uint8_t* gArena = nullptr;
static const tflite::Model* gModel = nullptr;
static tflite::MicroInterpreter* gInterp = nullptr;
static TfLiteTensor* gIn = nullptr;
static TfLiteTensor* gOut = nullptr;

bool initModel() {
  gModel = tflite::GetModel(g_model);
  if (gModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema %d != %d\n", (int)gModel->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  gArena = (uint8_t*)heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!gArena) { Serial.println("Không cấp phát được tensor arena (PSRAM?)"); return false; }

  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();

  static tflite::MicroInterpreter interp(gModel, resolver, gArena, kArenaSize);
  gInterp = &interp;
  if (gInterp->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors() lỗi"); return false; }

  gIn = gInterp->input(0);
  gOut = gInterp->output(0);
  if (gIn->type != kTfLiteInt8 || gIn->dims->size != 4 || gIn->dims->data[1] != IMG ||
      gIn->dims->data[2] != IMG || gIn->dims->data[3] != 1) {
    Serial.println("Input model không phải int8 1x96x96x1");
    return false;
  }
  if (kNumClasses > MAX_CLASSES) { Serial.println("Quá nhiều lớp"); return false; }
  Serial.printf("Model OK: %u bytes, arena dùng %u bytes\n", g_model_len,
                (unsigned)gInterp->arena_used_bytes());
  return true;
}

// Cắt vuông giữa ảnh xám -> trung bình vùng (giống INTER_AREA) -> 96x96 int8
static void preprocess(const camera_fb_t* fb, int8_t* dst) {
  const int w = fb->width, h = fb->height;
  const int side = (w < h) ? w : h;
  const int x0 = (w - side) / 2, y0 = (h - side) / 2;
  const float step = (float)side / IMG;
  const float inv = 1.0f / (255.0f * gIn->params.scale);
  const int zp = gIn->params.zero_point;
  // GRAYSCALE: 1 byte/pixel. YUV422 (YUYV): 2 byte/pixel, kênh Y nằm ở byte chẵn.
  const int bpp = (int)(fb->len / ((size_t)w * h)) >= 2 ? 2 : 1;

  for (int y = 0; y < IMG; y++) {
    int ys = (int)(y * step);
    int ye = (int)ceilf((y + 1) * step);
    if (ye > side) ye = side;
    if (ye <= ys) ye = ys + 1;
    for (int x = 0; x < IMG; x++) {
      int xs = (int)(x * step);
      int xe = (int)ceilf((x + 1) * step);
      if (xe > side) xe = side;
      if (xe <= xs) xe = xs + 1;
      uint32_t sum = 0;
      for (int yy = ys; yy < ye; yy++) {
        const uint8_t* row = fb->buf + ((size_t)(y0 + yy) * w + x0) * bpp;
        for (int xx = xs; xx < xe; xx++) sum += row[xx * bpp];
      }
      float v = ((float)sum / ((ye - ys) * (xe - xs))) * inv + zp;
      int q = lroundf(v);
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      dst[y * IMG + x] = (int8_t)q;
    }
  }
}

static void runInference(const camera_fb_t* fb) {
  preprocess(fb, gIn->data.int8);
  uint32_t t = millis();
  if (gInterp->Invoke() != kTfLiteOk) return;
  uint32_t dt = millis() - t;

  static float smooth[MAX_CLASSES] = {0};
  static int lastBest = -2;
  int best = 0;
  for (int i = 0; i < kNumClasses; i++) {
    float p = (gOut->data.int8[i] - gOut->params.zero_point) * gOut->params.scale;
    smooth[i] = 0.5f * smooth[i] + 0.5f * p;     // làm mượt nhẹ
    if (smooth[i] > smooth[best]) best = i;
  }
  int label = (smooth[best] >= CONF_THRESHOLD) ? best : -1;

  xSemaphoreTake(gMutex, portMAX_DELAY);
  for (int i = 0; i < kNumClasses; i++) gProbs[i] = smooth[i];
  gBest = label;
  gConf = smooth[best];
  gInferMs = dt;
  xSemaphoreGive(gMutex);

  if (label != lastBest) {
    lastBest = label;
    Serial.printf("Gesture: %-10s (%.0f%%)  infer=%ums\n",
                  label >= 0 ? kLabels[label] : "unknown", smooth[best] * 100, dt);
  }
}
#endif  // HAS_MODEL

// ---------- Camera ----------
bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_GRAYSCALE;   // 1 byte/pixel -> dễ tiền xử lý
  c.frame_size = FRAMESIZE_QVGA;          // 320x240
  c.jpeg_quality = 12;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    // Cam GC2145 không có JPEG/xám gốc -> thử YUV422 (lấy kênh Y làm ảnh xám)
    Serial.printf("Init GRAYSCALE lỗi 0x%x, thử YUV422...\n", err);
    esp_camera_deinit();
    c.pixel_format = PIXFORMAT_YUV422;
    err = esp_camera_init(&c);
  }
  if (err != ESP_OK) {
    Serial.printf("Camera init lỗi 0x%x (kiểm tra chân camera / cáp FPC)\n", err);
    return false;
  }
  Serial.printf("Camera OK, định dạng: %s\n",
                c.pixel_format == PIXFORMAT_GRAYSCALE ? "GRAYSCALE" : "YUV422");

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    // Nếu hình bị ngược/gương thì đổi 2 dòng dưới (nhớ thu dataset cùng chiều):
    // s->set_vflip(s, 1);
    // s->set_hmirror(s, 1);
  }
  return true;
}

// ---------- Task: chụp -> nhận dạng -> nén JPEG ----------
void aiTask(void*) {
  for (;;) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { vTaskDelay(10); continue; }

#if HAS_MODEL
    runInference(fb);
#endif

    uint8_t* out = nullptr;
    size_t olen = 0;
    bool ok = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, fb->format, 60, &out, &olen);
    esp_camera_fb_return(fb);

    if (ok) {
      xSemaphoreTake(gMutex, portMAX_DELAY);
      if (gJpg) free(gJpg);
      gJpg = out;
      gJpgLen = olen;
      gFrameId++;
      xSemaphoreGive(gMutex);
    }
    vTaskDelay(1);
  }
}

// ---------- Web ----------
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32-S3 Gesture</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;text-align:center;margin:0;padding:12px}
.w{position:relative;display:inline-block;width:min(94vw,480px)}
.w img{width:100%;border-radius:8px;display:block}
.sq{position:absolute;left:12.5%;width:75%;top:0;bottom:0;border:2px dashed #0f0;box-sizing:border-box;pointer-events:none}
#lb{font-size:2.4em;margin:.3em 0 0}#sub{color:#999;margin-bottom:10px}
.r{display:flex;align-items:center;max-width:480px;margin:3px auto;gap:8px}
.r span{width:90px;text-align:left}.r div{flex:1;background:#333;height:14px;border-radius:7px;overflow:hidden}
.r i{display:block;height:100%;background:#4caf50;width:0}
</style></head><body>
<div class="w"><img id="v"><div class="sq"></div></div>
<div id="lb">...</div><div id="sub"></div><div id="bars"></div>
<p style="color:#777;font-size:.85em">Đặt bàn tay trong khung xanh (vùng AI nhìn thấy)</p>
<script>
document.getElementById('v').src='http://'+location.hostname+':81/stream';
async function t(){
  try{
    const j=await (await fetch('/status')).json();
    document.getElementById('lb').textContent=j.label;
    document.getElementById('sub').textContent=
      j.classes.length?('độ tin cậy '+(j.conf*100).toFixed(0)+'% | '+j.ms+' ms/khung'):'Chưa có model - chỉ stream cam';
    let h='';
    j.classes.forEach((c,i)=>{h+='<div class="r"><span>'+c+'</span><div><i style="width:'+(j.probs[i]*100).toFixed(0)+'%"></i></div></div>';});
    document.getElementById('bars').innerHTML=h;
  }catch(e){}
  setTimeout(t,250);
}
t();
</script></body></html>
)rawliteral";

static esp_err_t indexHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t statusHandler(httpd_req_t* req) {
  String j = "{";
#if HAS_MODEL
  xSemaphoreTake(gMutex, portMAX_DELAY);
  j += "\"label\":\"" + String(gBest >= 0 ? kLabels[gBest] : "unknown") + "\"";
  j += ",\"conf\":" + String(gConf, 2);
  j += ",\"ms\":" + String(gInferMs);
  j += ",\"classes\":[";
  for (int i = 0; i < kNumClasses; i++) { if (i) j += ","; j += "\"" + String(kLabels[i]) + "\""; }
  j += "],\"probs\":[";
  for (int i = 0; i < kNumClasses; i++) { if (i) j += ","; j += String(gProbs[i], 2); }
  j += "]";
  xSemaphoreGive(gMutex);
#else
  j += "\"label\":\"(no model)\",\"conf\":0,\"ms\":0,\"classes\":[],\"probs\":[]";
#endif
  j += "}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, j.c_str(), j.length());
}

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CT = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BD = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t streamHandler(httpd_req_t* req) {
  const size_t CAP = 80 * 1024;
  uint8_t* buf = (uint8_t*)ps_malloc(CAP);
  if (!buf) return ESP_FAIL;

  httpd_resp_set_type(req, STREAM_CT);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  uint32_t lastId = 0;
  esp_err_t res = ESP_OK;

  while (res == ESP_OK) {
    size_t len = 0;
    xSemaphoreTake(gMutex, portMAX_DELAY);
    if (gJpg && gFrameId != lastId && gJpgLen <= CAP) {
      len = gJpgLen;
      memcpy(buf, gJpg, len);
      lastId = gFrameId;
    }
    xSemaphoreGive(gMutex);

    if (!len) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

    char part[64];
    size_t hl = snprintf(part, sizeof(part), STREAM_PART, (unsigned)len);
    res = httpd_resp_send_chunk(req, STREAM_BD, strlen(STREAM_BD));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part, hl);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)buf, len);
  }
  free(buf);
  return res;
}

void startWebServers() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;
  httpd_handle_t web = nullptr;
  httpd_uri_t uIndex = {"/", HTTP_GET, indexHandler, nullptr};
  httpd_uri_t uStatus = {"/status", HTTP_GET, statusHandler, nullptr};
  if (httpd_start(&web, &cfg) == ESP_OK) {
    httpd_register_uri_handler(web, &uIndex);
    httpd_register_uri_handler(web, &uStatus);
  }

  cfg.server_port = 81;
  cfg.ctrl_port += 1;
  cfg.stack_size = 8192;
  httpd_handle_t stream = nullptr;
  httpd_uri_t uStream = {"/stream", HTTP_GET, streamHandler, nullptr};
  if (httpd_start(&stream, &cfg) == ESP_OK) httpd_register_uri_handler(stream, &uStream);
}

// ---------- WiFi: AP cố định để IP luôn là 192.168.4.1 ----------
void startWiFi() {
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  IPAddress ip(192, 168, 4, 1);
  IPAddress gateway(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  if (!WiFi.softAPConfig(ip, gateway, subnet)) {
    Serial.println("Không đặt được IP AP cố định");
  }
  if (!WiFi.softAP(AP_SSID, AP_PASS)) {
    Serial.println("Không khởi động được WiFi AP");
  } else {
    Serial.printf("AP sẵn sàng: %s\n", AP_SSID);
  }
}

void printAddress() {
  Serial.println("=====================================");
  Serial.printf("AP: %s / %s\n", AP_SSID, AP_PASS);
  Serial.println("Mở:  http://192.168.4.1");
  Serial.println("Stream: http://192.168.4.1:81/stream");
  Serial.println("=====================================");
}

// ---------- setup / loop ----------
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nESP32-S3 Gesture Recognition");

  if (!psramFound()) Serial.println("CẢNH BÁO: không thấy PSRAM - kiểm tra Tools > PSRAM!");

  gMutex = xSemaphoreCreateMutex();

  if (!initCamera()) {
    Serial.println("Dừng: camera lỗi");
    startWiFi();
    printAddress();
    while (true) delay(1000);
  }

#if HAS_MODEL
  if (!initModel()) { Serial.println("Dừng: model lỗi"); while (true) delay(1000); }
#else
  Serial.println("Chưa có model_data.h -> chế độ chỉ stream cam");
#endif

  startWiFi();
  startWebServers();
  printAddress();

  xTaskCreatePinnedToCore(aiTask, "ai", 16384, nullptr, 2, nullptr, 1);
}

void loop() {
  static uint32_t t = 0;
  if (millis() - t > 5000) {   // in lại IP định kỳ để bạn mở Serial Monitor lúc nào cũng thấy
    t = millis();
    printAddress();
  }
  delay(100);
}
