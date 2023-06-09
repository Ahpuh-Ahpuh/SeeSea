//데모영상, seesea 종료 등 버튼 있는 페이지

package com.example.ss0327;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttp;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class Start_page extends AppCompatActivity {
    private Retrofit mRetrofit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start_page);
        Button setbtn = (Button) findViewById(R.id.setbtn);
        //Button linkbtn = (Button) findViewById(R.id.linkbtn);
        Button endbtn = (Button) findViewById(R.id.endbtn);
        Button startbtn = (Button) findViewById(R.id.startbtn);
        // 데모 버튼 이거임
        Button startdemo = (Button) findViewById(R.id.startdemo);
        //setRetrofitInit();
        setbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), SettingActivity.class);
                startActivity(intent);
            }
        });
        //demo 버튼 누르면 데모 액티비티~
        startdemo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), DemoActivity.class);
                startActivity(intent);
            }
        });
        /*
        linkbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), Link_page.class);
                startActivity(intent);
            }
        });*/
        endbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finishAffinity();
                System.exit(0);
            }
        });
        startbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //감시 시작 버튼 -> mainactivity로 넘어감
                Intent intent = new Intent(Start_page.this, DroneActivity.class);
                startActivity(intent);
            }
        });
    }
}