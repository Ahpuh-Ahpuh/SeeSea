package com.example.ss0327;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageButton;

public class third_page extends AppCompatActivity {
    Button button;
    ImageButton imgbtn;
    third_page tp=null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_third_page);
        tp = this;


        ImageButton btn_dialog = (ImageButton) findViewById(R.id.imageButton4);
        /*ImageView img_result;
        img_result=(ImageView) findViewById(R.id.img_result);
        img_result.setVisibility(View.INVISIBLE);
*/
        btn_dialog.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showA(v);
            }
        });
    }
    public void showA(View _view){
        alarm Alarm = new alarm(this);
        Alarm.setCancelable(false);
        Alarm.show();
    }
}