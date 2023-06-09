package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.VideoView;

public class MainActivity2 extends AppCompatActivity {
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
//        ImageButton ib1=(ImageButton) findViewById(R.id.ib1);
//        ImageButton ib2=(ImageButton) findViewById(R.id.ib2);
//        ImageButton ib3=(ImageButton) findViewById(R.id.ib3);
////        ib1.setOnClickListener(new View.OnClickListener() {
////            @Override
////            public void onClick(View view) {
////                Intent intent = new Intent(getApplicationContext(),Second_page2.class);
////                startActivity(intent);
////            }
////        });
//        ib2.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent = new Intent(getApplicationContext(),third_page.class);
//                startActivity(intent);
//            }
//        });
//        ib3.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent = new Intent(getApplicationContext(),fourth_page.class);
//                startActivity(intent);
//            }
//        });
        VideoView videoView = findViewById(R.id.videoView2);
        String videoPath = "android.resource://" + getPackageName() + "/" + R.raw.data11; // Replace "my_video" with the actual video file name
        videoView.setVideoPath(videoPath);
        videoView.start(); // Start playing the video
//        videoView.pause(); // Pause the video
//        videoView.seekTo(5000); // Seek to a specific position (in milliseconds)
//        videoView.stopPlayback(); // Stop playback and release resources
        Button btn=(Button) findViewById(R.id.fbtn);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                /*Intent intent=new Intent(getApplicationContext(),Start_page.class);
                startActivity(intent);*/
                // start page로 다시 돌아감
                onBackPressed();
            }
        });
    }
}