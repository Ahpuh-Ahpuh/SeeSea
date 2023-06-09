//package com.example.ss0327;
//
//import android.os.AsyncTask;
//import android.os.Bundle;
//import android.util.Log;
//import android.widget.ImageView;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import com.bumptech.glide.Glide;
//
//import java.io.IOException;
//import java.net.HttpURLConnection;
//import java.net.URL;
//
//public class demo_tmp extends AppCompatActivity {
////    @Override
////    protected void onCreate(Bundle savedInstanceState) {
////        super.onCreate(savedInstanceState);
////        binding = ActivityDemoBinding.inflate(getLayoutInflater())
////        setContentView(R.layout.activity_demo);
////    }
////
//    ImageView imgGlide;
//
//    private static final String TAG = "MainActivity";
//    private static final String API_URL = "http://10.0.2.2:5000/video/"; // Replace with your API URL
////    String API_URL = "C:\\SeeSea\\front_new\\ss0327\\app\\src\\main\\java\\com\\example\\ss0327\\cat.jpg";
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        Log.d("here","------------------------------------------------");
//        Log.e("here",API_URL);
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_demo);
//
//
////        DemoActivity obj = new DemoActivity();
////        @SuppressWarnings("rawtypes")
////        Class resource = obj.getClass();
////        URL imageurl = resource.getResource("http://10.0.2.2:5000/video");
////        System.out.println("Resource URL one is = " + imageurl);
//
////        imgGlide = findViewById(R.id.imageView3);
////        Glide.with(this).load(R.drawable.cat).into(imgGlide);
////        Log.e("here",API_URL);
////        ImageView imageView = findViewById(R.id.imageView3);
////
////        String imageUrl = "http://10.0.2.2:5000/video";  // Replace with the appropriate URL of your Flask server
////
////// Using Glide to load and display the image
//        Glide.with(this)
//                .load(imageUrl)
//                .into(imageView);
//
//    }
//
//    private class ApiCallTask extends AsyncTask<String, Void, String> {
//
//        @Override
//        protected String doInBackground(String... urls) {
//            String apiUrl = urls[0];
//            String response = "";
//
//            try {
//                URL url = new URL(apiUrl);
//                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
//                connection.setRequestMethod("GET");
//
//                imgGlide = findViewById(R.id.imageView3);
//
////                Glide.with(this).load(img).into(imgGlide);
//                // Get the response
////                InputStream inputStream = connection.getInputStream();
////                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
////                StringBuilder stringBuilder = new StringBuilder();
////                String line;
////                while ((line = reader.readLine()) != null) {
////                    stringBuilder.append(line);
////                }
////                response = stringBuilder.toString();
//
//                // Close the connections
////                reader.close();
////                inputStream.close();
////                connection.disconnect();
//            } catch (IOException e) {
//                Log.e(TAG, "Error occurred during API call: " + e.getMessage());
//            }
//
//            return response;
//        }
//
//        @Override
//        protected void onPostExecute(String response) {
//            // Handle the API response here
//            Log.d(TAG, "API response: " + response);
//        }
//    }
//}