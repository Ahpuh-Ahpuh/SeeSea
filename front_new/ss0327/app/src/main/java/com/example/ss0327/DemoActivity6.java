//package com.example.ss0327;
//
//import android.content.Intent;
//import android.graphics.Bitmap;
//import android.graphics.BitmapFactory;
//import android.os.Bundle;
//import android.os.Handler;
//import android.os.Looper;
//import android.util.Log;
//import android.view.MotionEvent;
//import android.view.View;
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
//public class DemoActivity extends AppCompatActivity {
//    private WebView webView;
//    private final OkHttpClient client = new OkHttpClient();
//    private Handler handler;
//    private final int delay = 200; //milliseconds
//    String url1 = "http://10.0.2.2:8000/alarm";
//    String url2 = "http://10.0.2.2:8000/seen";
//    String url3 = "http://10.0.2.2:8000/depth"; // new endpoint for depth
//    private int url1Counter = 0; // Counter for url1
//    private int url2Counter = 0; // Counter for url2
//    int flag = 0;
//    alarm AlarmDialog;
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//        AlarmDialog = new alarm(this);
//        AlarmDialog.setCancelable(true);
//        handler = new Handler(Looper.getMainLooper());
//        webView = (WebView) findViewById(R.id.webview);
//        webView.setWebViewClient(new WebViewClient() {
//            public void onPageFinished(WebView view, String url) {
//                // Handler for repeating tasks
//
//                // Runnable for HTTP request for first URL
//                Runnable runnableCode1 = getRunnableForUrl(url1);
//                handler.post(runnableCode1);
//            }
//        });
//
//        WebSettings webSettings = webView.getSettings();
//        webSettings.setJavaScriptEnabled(true);
//        webView.loadUrl("http://10.0.2.2:8000/video");
//
//        WebView webView2 = (WebView) findViewById(R.id.webview2);
//        webView2.setWebViewClient(new WebViewClient() {
//            public void onPageFinished(WebView view, String url) {
//                // Handler for repeating tasks
//
//                // Runnable for HTTP request for second URL
//                Runnable runnableCode2 = getRunnableForUrl(url2);
//                handler.post(runnableCode2);
//            }
//        });
//
//        WebSettings webSettings2 = webView2.getSettings();
//        webSettings2.setJavaScriptEnabled(true);
//        webView2.loadUrl("http://10.0.2.2:8000/normal1");
//
//        WebView webView3 = (WebView) findViewById(R.id.webview3);
//        webView3.setWebViewClient(new WebViewClient() {
//            public void onPageFinished(WebView view, String url) {
//                // Handler for repeating tasks
//
//                // Runnable for HTTP request for third URL
//                Runnable runnableCode3 = getRunnableForUrl(url3);
//                handler.post(runnableCode3);
//            }
//        });
//
//        WebSettings webSettings3 = webView3.getSettings();
//        webSettings3.setJavaScriptEnabled(true);
//        webView3.loadUrl("http://10.0.2.2:8000/normal2");
//    }
//
//    @Override
//    protected void onResume() {
//        super.onResume();
//        // Start or restart the runnable tasks by posting through the handler
//        if (handler != null) {
//            handler.post(getRunnableForUrl(url1));
//            handler.post(getRunnableForUrl(url2));
//            handler.post(getRunnableForUrl(url3));
//        }
//    }
//
//    @Override
//    protected void onPause() {
//        super.onPause();
//        if (AlarmDialog != null && AlarmDialog.isShowing()) {
//            AlarmDialog.dismiss();
//        }
//
//        // Cancel HTTP requests when activity is not in the foreground
//        handler.removeCallbacksAndMessages(null);
//    }
////    @Override
////    protected void onDestroy() {
////        super.onDestroy();
////        // Remove any pending posts of Runnable r that are in the message queue when activity is destroyed
////        handler.removeCallbacksAndMessages(null);
////    }
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
//                                        if (url.endsWith("/alarm") && responseData[0].equals("1") && flag == 0) {
//                                            // Request for the depth data
//                                            Request requestDepth = new Request.Builder()
//                                                    .url(url3)
//                                                    .build();
//                                            client.newCall(requestDepth).enqueue(new Callback() {
//                                                @Override
//                                                public void onFailure(Call call, IOException e) {
//                                                    e.printStackTrace();
//                                                }
//
//                                                @Override
//                                                public void onResponse(Call call, Response response) throws IOException {
//                                                    if (response.isSuccessful()) {
//                                                        final String depth = response.body().string();
//                                                        new Handler(Looper.getMainLooper()).post(new Runnable() {
//                                                            @Override
//                                                            public void run() {
//
//                                                                AlarmDialog.setDistance(depth); // set depth as message
//                                                                AlarmDialog.show();
//                                                                flag = 1;
//                                                            }
//                                                        });
//                                                    }
//                                                }
//                                            });
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
////