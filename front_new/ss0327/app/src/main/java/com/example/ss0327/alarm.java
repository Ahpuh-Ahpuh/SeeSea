package com.example.ss0327;

import android.app.Dialog;
import android.content.Context;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.VideoView;

import androidx.annotation.NonNull;
public class alarm extends Dialog {
    MediaPlayer mediaPlayer;
    alarm Alarm;
    VideoView videoView;
    TextView dist;

    public alarm(Context context){
        super(context, android.R.style.Theme_Translucent_NoTitleBar);
    }
    public alarm(){
        super(null);
    }
    @Override
    public void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        WindowManager.LayoutParams IpWindow = new WindowManager.LayoutParams();
        IpWindow.flags=WindowManager.LayoutParams.FLAG_DIM_BEHIND;
        IpWindow.dimAmount=0.5f;
        getWindow().setAttributes(IpWindow);

        setContentView(R.layout.activity_alarm);
        dist = (TextView) findViewById(R.id.textView2);
        mediaPlayer = MediaPlayer.create(getContext(),R.raw.siren_sound);
        mediaPlayer.setLooping(true);
        Alarm=this;

        TextView tv=(TextView) this.findViewById(R.id.textView);
        tv.setText("!!Warning!!");

        WebView webView = findViewById(R.id.webview1);
        webView.loadUrl("http://10.0.2.2:8000/alarm_page");
        WebSettings webSettings = webView.getSettings();
        webSettings.setUseWideViewPort(true);
        webSettings.setLoadWithOverviewMode(true);
        
        Button bt=(Button) this.findViewById(R.id.btn_shutdown);
        bt.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                onClickBtn(v);
            }
        });

        mediaPlayer.start();
    }

    public void setDistance(String distance) {
        if (dist != null) {
            dist.setText(distance);
        }
    }

    public void dismiss(){
        super.dismiss();
        mediaPlayer.release();
    }

    public void onClickBtn(View _oView){
        this.dismiss();
    }
}
