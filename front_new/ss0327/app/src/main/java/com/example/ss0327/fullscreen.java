package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;

import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.widget.MediaController;
import android.widget.VideoView;

public class fullscreen extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_fullscreen);
        VideoView videoView = findViewById(R.id.videoView);
        String videoPath = "android.resource://" + getPackageName() + "/" + R.raw.data11; // Replace video_file with the name of your video file in the res/raw directory
        videoView.setVideoURI(Uri.parse(videoPath));
        videoView.setRotation(180);
        MediaController mediaController = new MediaController(this);
        videoView.setMediaController(mediaController);

                videoView.start();
    }
}