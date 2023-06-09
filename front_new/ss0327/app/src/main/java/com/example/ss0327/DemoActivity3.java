//package com.example.ss0327;
//
//import android.content.Intent;
//import android.graphics.Bitmap;
//import android.graphics.BitmapFactory;
//import android.os.Bundle;
//import android.os.Handler;
//import android.os.Looper;
//import android.util.Log;
//import android.webkit.WebSettings;
//import android.webkit.WebView;
//import android.webkit.WebViewClient;
//import android.widget.ImageView;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import java.io.ByteArrayOutputStream;
//import java.io.IOException;
//import java.io.InputStream;
//import java.util.Arrays;
//
//import okhttp3.Call;
//import okhttp3.Callback;
//import okhttp3.OkHttpClient;
//import okhttp3.Request;
//import okhttp3.Response;
//
//public class DemoActivity extends AppCompatActivity {
//    private WebView webView;
//    private final OkHttpClient client = new OkHttpClient();
//    private Handler handler;
//    private final int delay = 200; //milliseconds
//    String url1 = "http://10.0.2.2:8000/alarm";
//    String url2 = "http://10.0.2.2:8000/seen";
//    private int url1Counter = 0; // Counter for url1
//    private int url2Counter = 0; // Counter for url2
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//
//        webView = (WebView) findViewById(R.id.webview);
//        webView.setWebViewClient(new WebViewClient()); // uses WebViewClient to handle webpage changes
//
//        // Enable JavaScript (necessary for some video streaming)
//        WebSettings webSettings = webView.getSettings();
//        webSettings.setJavaScriptEnabled(true);
//
//        // Load your video streaming URL
//        webView.loadUrl("http://10.0.2.2:8000/video"); // replace with your FastAPI server's IP and port
//
//        String url1 = "http://10.0.2.2:8000/alarm";
//        String url2 = "http://10.0.2.2:8000/seen";
//
//        // Handler for repeating tasks
//        handler = new Handler(Looper.getMainLooper());
//
//        // Runnable for HTTP request for first URL
//        Runnable runnableCode1 = getRunnableForUrl(url1);
//        // Runnable for HTTP request for second URL
//        Runnable runnableCode2 = getRunnableForUrl(url2);
//
//        // Start the initial runnable tasks by posting through the handler
//        handler.post(runnableCode1);
//        handler.post(runnableCode2);
//    }
//
//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        // Remove any pending posts of Runnable r that are in the message queue when activity is destroyed
//        handler.removeCallbacksAndMessages(null);
//    }
//
//    private Runnable getRunnableForUrl(final String url) {
//        return new Runnable() {
//            @Override
//            public void run() {
//
//                // Build the request
//                Request request = new Request.Builder()
//                        .url(url)
//                        .build();
//
//                // Enqueue the request (asynchronous call)
//                client.newCall(request).enqueue(new Callback() {
//                    @Override
//                    public void onFailure(Call call, IOException e) {
//                        e.printStackTrace();
//                    }
//
//                    @Override
//                    public void onResponse(Call call, final Response response) throws IOException {
//                        if (!response.isSuccessful()) {
//                            throw new IOException("Unexpected code " + response);
//                        } else {
//                            try {
//                                final String[] responseData = {response.body().string()};
//
//                                if (url.equals(url1) && url1Counter++ < 10) {
//                                    responseData[0] = "0";
//                                }
//
//                                if (url.equals(url2) && url2Counter++ < 10) {
//                                    responseData[0] = "0";
//                                }
//
//                                new Handler(Looper.getMainLooper()).post(new Runnable() {
//                                    @Override
//                                    public void run() {
//                                        Log.i("OkHttp", "URL: " + url + ", Data: " + responseData[0]);
//                                        if (url.endsWith("/alarm") && responseData[0].equals("1")) {
//                                            alarm AlarmDialog = new alarm(DemoActivity.this);
//                                            AlarmDialog.show();
//                                        }
//                                    }
//                                });
//
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    }
//
//                });
//
//                // Repeat this runnable code again every 5 seconds
//                handler.postDelayed(this, delay);
//            }
//        };
//    }
//}
