package com.example.ss0327;

import android.content.Intent;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class DroneActivity extends AppCompatActivity {
    private WebView webView;
    private final OkHttpClient client = new OkHttpClient();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_drone);

        webView = findViewById(R.id.webview);
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                if (url.equals("http://10.0.2.2:8000/land")) {
                    return false; // Allow loading the specified URL
                } else {
                    return true; // Prevent loading other URLs
                }
            }
        });

        // Enable JavaScript (necessary for some video streaming)
        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);

        // Load your video streaming URL
        webView.loadUrl("http://10.0.2.2:8000/land"); // replace with your FastAPI server's IP and port

        webView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    Intent intent = new Intent(DroneActivity.this, FullDemo.class);
                    startActivity(intent);
                }
                return false;
            }
        });

        Button stopDroneBtn = findViewById(R.id.stopDroneBtn);
        stopDroneBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Send an HTTP GET request to the /stop endpoint
                Request request = new Request.Builder()
                        .url("http://10.0.2.2:8000/stop")
                        .build();

                client.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(Call call, IOException e) {
                        e.printStackTrace();
                    }

                    @Override
                    public void onResponse(Call call, final Response response) throws IOException {
                        if (!response.isSuccessful()) {
                            throw new IOException("Unexpected code " + response);
                        } else {
                            // Stop the WebView from loading further
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    webView.stopLoading();
                                    webView.loadUrl("about:blank");
                                }
                            });
                        }
                    }
                });
            }
        });
    }

//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        // Load a blank URL before destroying the WebView
//        webView.loadUrl("about:blank");
//        // Destroy the WebView to prevent potential memory leaks
//        webView.destroy();
//    }
}
