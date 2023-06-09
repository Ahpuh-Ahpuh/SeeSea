// 프레임 안받아와지는 코드
//    package com.example.ss0327;
//
//    import android.graphics.Bitmap;
//    import android.graphics.BitmapFactory;
//    import android.os.Bundle;
//    import android.os.Handler;
//    import android.os.Looper;
//    import android.util.Log;
//    import android.widget.ImageView;
//
//    import androidx.appcompat.app.AppCompatActivity;
//
//    import java.io.ByteArrayOutputStream;
//    import java.io.IOException;
//    import java.io.InputStream;
//    import java.util.Arrays;
//
//    import okhttp3.Call;
//    import okhttp3.Callback;
//    import okhttp3.OkHttpClient;
//    import okhttp3.Request;
//    import okhttp3.Response;
//
//    public class DemoActivity4 extends AppCompatActivity {
//        private OkHttpClient client;
//        private Handler handler;
//        private ImageView imageView;
//
//        @Override
//        protected void onCreate(Bundle savedInstanceState) {
//            super.onCreate(savedInstanceState);
//            setContentView(R.layout.activity_demo);
//
//            // ImageView 초기화
//            imageView = findViewById(R.id.imageView3);
//
//            // OkHttpClient 및 Handler 초기화
//            client = new OkHttpClient();
//            handler = new Handler(Looper.getMainLooper());
//
//            // 프레임 스트리밍 시작
//            startFrameStreaming();
//        }
//
//        private void startFrameStreaming() {
//            // Flask 서버의 /video 엔드포인트에 대한 요청 생성
//            Request request = new Request.Builder()
//                    .url("http://10.0.2.2:8000/video")
//                    .build();
//
//            // 비동기적으로 요청 보내기
//            client.newCall(request).enqueue(new Callback() {
//                @Override
//                public void onResponse(Call call, Response response) throws IOException {
//                    // 응답을 받았을 때 실행되는 코드
//                    Log.e("55 l", "here");
//                    if (response.isSuccessful()) {
//                        InputStream inputStream = response.body().byteStream();
//                        //여기까지 넘어옴
//                        System.out.println("here "+inputStream);
//                        // 프레임 데이터를 읽고 ImageView에 표시하는 루프 실행
//                        while (true) {
//                            Log.e("63", "63 printtt!!!!");
//                            // 프레임 데이터 읽기
//                            byte[] frameData = readFrame(inputStream);
//                            Log.e("63", "-----------------!!!!");
//                            if (frameData != null) {
//                                // 프레임 데이터를 Bitmap으로 디코딩하여 ImageView에 업데이트
//                                Log.e("66", "66 printtt!!!!");
//                                updateImageView(frameData);
//                            }
//
//                            // 필요한 경우 스레드를 지연시키는 코드 추가
//                            // 프레임 속도를 조절하려면 이 부분을 수정하십시오.
//                            try {
//                                Thread.sleep(100); // 100ms 지연
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    } else {
//                        Log.e("frame fail", "else");
//                    }
//                }
//
//                @Override
//                public void onFailure(Call call, IOException e) {
//                    Log.e("84frame fail", "else");
//                    // 요청이 실패한 경우 처리
//                }
//            });
//        }
//
//        private byte[] readFrame(InputStream inputStream) throws IOException {
//            // 프레임 데이터 읽기 및 완전성 확인 로직을 여기에 추가
//            // 앞서 제시한 `readFrame` 메서드를 사용하거나 프레임 데이터를 직접 처리하십시오.
//            byte[] buffer = new byte[8192]; // 필요에 따라 버퍼 크기를 조정하세요
//
//            // 프레임 데이터를 수집하기 위해 ByteArrayOutputStream을 생성합니다
//            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
//
//            // 입력 스트림에서 프레임 데이터를 읽습니다
//            int bytesRead;
//            while ((bytesRead = inputStream.read(buffer)) != -1) {
//                // 읽은 바이트를 ByteArrayOutputStream에 기록합니다
//                System.out.println("bytesRead"+bytesRead); // 잘 읽어오고잇음
//                byteArrayOutputStream.write(buffer, 0, bytesRead);
//
//                // 프레임 데이터가 완전한지 확인합니다
//                if (isFrameComplete(byteArrayOutputStream.toByteArray())) {
//                    // ByteArrayOutputStream을 바이트 배열로 변환합니다
//                    byte[] frameData = byteArrayOutputStream.toByteArray();
//                    System.out.println("framedata"+frameData);
//                    // 다음 프레임을 위해 ByteArrayOutputStream을 초기화합니다
//                    byteArrayOutputStream.reset();
//                    System.out.println("framedata 2 "+frameData);
//                    // 프레임 데이터를 반환합니다
//                    return frameData;
//                }
//            }
//
//            // 프레임 데이터가 완전하지 않거나 데이터가 없는 경우 null을 반환합니다
//            Log.e("fail", "here");
//            return null;
//        }
//
//        // 프레임 데이터가 완전한지 확인하는 메서드
//        private boolean isFrameComplete(byte[] frameData) {
//            final byte JPEG_START_MARKER = (byte) 0xFF;
//            final byte JPEG_END_MARKER = (byte) 0xD9;
//            Log.e("128", "aaaa-----------------!!!!");
//            System.out.println("array"+frameData[0] + "array2" + frameData[1] + "start "+ JPEG_START_MARKER);
//            System.out.println("출력합니다"+Arrays.toString(frameData)); //-> frameData 배열에 잘 저장되고 잇는거 확인함
//            // 프레임 데이터의 시작과 끝 마커 확인
//            boolean startsWithMarker = frameData.length > 1 && frameData[0] == JPEG_START_MARKER && frameData[1] == JPEG_START_MARKER;
//            boolean endsWithMarker = frameData.length > 1 && frameData[frameData.length - 2] == JPEG_END_MARKER && frameData[frameData.length - 1] == JPEG_START_MARKER;
//            System.out.println("le"+startsWithMarker);
//            System.out.println("end"+endsWithMarker);
//            // 프레임 데이터가 완전한지 여부 반환
//            return startsWithMarker && endsWithMarker;
//        }
//
//        private void updateImageView(byte[] frameData) {
//            // Bitmap으로 프레임 디코딩
//    //        Bitmap bitmap = BitmapFactory.decodeByteArray(frameData, 0, frameData.length);
//    //
//    //        // UI 스레드에서 ImageView 업데이트
//    //        handler.post(new Runnable() {
//    //            @Override
//    //            public void run() {
//    //                imageView.setImageBitmap(bitmap);
//    //            }
//
//            Bitmap bitmap = BitmapFactory.decodeByteArray(frameData, 0, frameData.length);
//
//            if (bitmap != null) {
//                // UI 스레드에서 ImageView 업데이트
//                handler.post(new Runnable() {
//                    @Override
//                    public void run() {
//                        Log.e("ccccc Activity", "Bitmap is null");
//                        imageView.setImageBitmap(bitmap);
//                    }
//                });
//            } else {
//                Log.e("DemoActivity", "Bitmap is null");
//            };
//        }
//
//    }
