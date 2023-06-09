package com.example.ss0327;
import java.io.Serializable;

public class Range implements Serializable {
    String x_range;
    String y_range;

    public Range (String x_range, String y_range){
        this.x_range=x_range;
        this.y_range=y_range;
    }
}


