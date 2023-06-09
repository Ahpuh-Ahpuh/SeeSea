package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;


public class SettingActivity extends AppCompatActivity {
    Button btn_main;
    EditText r1;
    EditText r2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setting_activity);
        btn_main=(Button) findViewById(R.id.button_main);
        r1=(EditText) findViewById(R.id.range1);
        r2=(EditText) findViewById(R.id.range2);

        btn_main.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String x_range=r1.getText().toString();
                String y_range=r2.getText().toString();
                try {
                    int x = Integer.parseInt(x_range);
                    int y = Integer.parseInt(y_range);

                    // Check if x and y are positive.
                    if (x > 0 && y > 0) {
                        // x_range and y_range are both positive integers.
                        Range range = new Range(x_range, y_range);
                        Intent intent = new Intent(getApplicationContext(), Drone1_page.class);
                        intent.putExtra("range", range);
                        startActivity(intent);
                    } else {
                        // At least one of x_range and y_range is not a positive integer.
                        throw new NumberFormatException();
                    }
                } catch (NumberFormatException e) {
                    // If we catch a NumberFormatException, then at least one of x_range and y_range is not a valid positive integer.
                    Custom2 customDialog = new Custom2(SettingActivity.this, "범위 입력");
                    customDialog.show();
                }
            }
        });
    }
}
