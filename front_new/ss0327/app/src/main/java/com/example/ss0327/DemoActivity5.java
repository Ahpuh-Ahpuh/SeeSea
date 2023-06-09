//package com.example.ss0327;
//import android.os.Bundle;
//import android.os.Handler;
//import android.os.Looper;
//import android.widget.ImageView;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import okhttp3.OkHttpClient;
//
//public class DemoActivity5 extends AppCompatActivity {
//    private OkHttpClient client;
//    private Handler handler;
//    private ImageView imageView;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//
//        // ImageView 초기화
//        imageView = findViewById(R.id.imageView3);
//
//        // OkHttpClient 및 Handler 초기화
//        client = new OkHttpClient();
//        handler = new Handler(Looper.getMainLooper());
//
//        // 프레임 스트리밍 시작
////        startFrameStreaming();
//    }
////
////    private void startFrameStreaming() {
////        // Flask 서버의 /video 엔드포인트에 대한 요청 생성
////        Request request = new Request.Builder()
////                .url("http://localhost:8000/video")
////                .build();
////
////        // 비동기적으로 요청 보내기
////        client.newCall(request).enqueue(new Callback() {
////            @Override
////            public void onResponse(Call call, Response response) throws IOException {
////                // 응답을 받았을 때 실행되는 코드
////                if (response.isSuccessful()) {
////                    InputStream inputStream = response.body().byteStream();
////
////                    // 프레임 데이터를 읽고 ImageView에 표시하는 루프 실행
////                    while (true) {
////                        // 프레임 데이터 읽기
////                        byte[] frameData = readFrame(inputStream);
////
////                        if (frameData != null) {
////                            // 프레임 데이터를 Bitmap으로 디코딩하여 ImageView에 업데이트
////                            updateImageView(frameData);
////                        }
////
////                        // 필요한 경우 스레드를 지연시키는 코드 추가
////                        // 프레임 속도를 조절하려면 이 부분을 수정하십시오.
////                        try {
////                            Thread.sleep(100); // 100ms 지연
////                        } catch (InterruptedException e) {
////                            e.printStackTrace();
////                        }
////                    }
////                } else {
////                    // 요청이 실패한 경우 처리
////                }
////            }
////
////            @Override
////            public void onFailure(Call call, IOException e) {
////                // 요청이 실패한 경우 처리
////            }
////        });
////    }
////
////    private byte[] readFrame(InputStream inputStream) throws IOException {
////        // 프레임 데이터 읽기 및 완전성 확인 로직을 여기에 추가
////        // 앞서 제시한 `readFrame` 메서드를 사용하거나 프레임 데이터를 직접 처리하십시오.
////        byte[] buffer = new byte[8192]; // 필요에 따라 버퍼 크기를 조정하세요
////
////        // 프레임 데이터를 수집하기 위해 ByteArrayOutputStream을 생성합니다
////        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
////
////        // 입력 스트림에서 프레임 데이터를 읽습니다
////        int bytesRead;
////        while ((bytesRead = inputStream.read(buffer)) != -1) {
////            // 읽은 바이트를 ByteArrayOutputStream에 기록합니다
////            byteArrayOutputStream.write(buffer, 0, bytesRead);
////
////            // 프레임 데이터가 완전한지 확인합니다
////            if (isFrameComplete(byteArrayOutputStream.toByteArray())) {
////                // ByteArrayOutputStream을 바이트 배열로 변환합니다
////                byte[] frameData = byteArrayOutputStream.toByteArray();
////
////                // 다음 프레임을 위해 ByteArrayOutputStream을 초기화합니다
////                byteArrayOutputStream.reset();
////
////                // 프레임 데이터를 반환합니다
////                return frameData;
////            }
////        }
////
////        // 프레임 데이터가 완전하지 않거나 데이터가 없는 경우 null을 반환합니다
////        return null;
////    }
////
////    // 프레임 데이터가 완전한지 확인하는 메서드
////    private boolean isFrameComplete(byte[] frameData) {
////        final byte JPEG_START_MARKER = (byte) 0xFF;
////        final byte JPEG_END_MARKER = (byte) 0xD9;
////
////        // 프레임 데이터의 시작과 끝 마커 확인
////        boolean startsWithMarker = frameData.length > 1 && frameData[0] == JPEG_START_MARKER && frameData[1] == JPEG_START_MARKER;
////        boolean endsWithMarker = frameData.length > 1 && frameData[frameData.length - 2] == JPEG_END_MARKER && frameData[frameData.length - 1] == JPEG_START_MARKER;
////
////        // 프레임 데이터가 완전한지 여부 반환
////        return startsWithMarker && endsWithMarker;
////    }
////
////    private void updateImageView(byte[] frameData) {
////        // Bitmap으로 프레임 디코딩
////        Bitmap bitmap = BitmapFactory.decodeByteArray(frameData, 0, frameData.length);
////
////        if (bitmap != null) {
////            // UI 스레드에서 ImageView 업데이트
////            handler.post(new Runnable() {
////                @Override
////                public void run() {
////                    imageView.setImageBitmap(bitmap);
////                }
////            });
////        }
////    }
//}
