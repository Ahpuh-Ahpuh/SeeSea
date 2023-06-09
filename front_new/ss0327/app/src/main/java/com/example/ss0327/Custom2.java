package com.example.ss0327;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import android.os.Bundle;
import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

public class Custom2 extends Dialog {

    private String message;

    public Custom2(Context context, String message) {
        super(context);
        this.message = message;
    }
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowManager.LayoutParams IpWindow = new WindowManager.LayoutParams();
        IpWindow.flags=WindowManager.LayoutParams.FLAG_DIM_BEHIND;
        IpWindow.dimAmount=0.5f;
        getWindow().setAttributes(IpWindow);
        requestWindowFeature(Window.FEATURE_NO_TITLE); // 타이틀 바 제거
        setContentView(R.layout.activity_custom2);

        TextView messageTextView =findViewById(R.id.message_text2);
        messageTextView.setText("         범위가 입력되지 않았거나 \n      유효하지 않은 값입니다\n    다시 입력해주세요.");

        Button bt=(Button) this.findViewById(R.id.btn_check2);
        bt.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                onClickBtn(v);
            }
        });

    }
    public void dismiss(){
        super.dismiss();
    }
    public void onClickBtn(View _oView){
        this.dismiss();
    }
}
