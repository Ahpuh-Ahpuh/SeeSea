package com.example.ss0327;

import android.os.Bundle;
import android.content.Intent;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;

public class IntroActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.intro);

        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent intent = new Intent(getApplicationContext(),Start_page.class);
                startActivity(intent);
                finish();
            }
        },3000);
    }
    @Override
    protected void onPause(){
        super.onPause();
        finish();
    }
}