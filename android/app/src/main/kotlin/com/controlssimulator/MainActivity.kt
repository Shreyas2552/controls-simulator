package com.controlssimulator

import android.annotation.SuppressLint
import android.os.Build
import android.os.Bundle
import android.webkit.*
import android.view.KeyEvent
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
        super.onCreate(savedInstanceState)

        // Make status bar and navigation bar match dark theme
        window.statusBarColor = 0xFF0E1117.toInt()
        window.navigationBarColor = 0xFF0E1117.toInt()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            window.decorView.systemUiVisibility =
                window.decorView.systemUiVisibility and View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR.inv()
        }

        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)

        // Prevent Android from applying algorithmic dark mode on top of our custom dark CSS
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            @Suppress("DEPRECATION")
            webView.settings.forceDark = WebSettings.FORCE_DARK_OFF
        }

        webView.settings.apply {
            javaScriptEnabled                    = true
            domStorageEnabled                    = true
            allowFileAccess                      = true
            allowFileAccessFromFileURLs          = true
            allowUniversalAccessFromFileURLs     = true
            mixedContentMode                     = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            loadWithOverviewMode                 = true
            useWideViewPort                      = true
            // Enable zoom so that Android passes pinch events through to JavaScript (Plotly)
            // builtInZoomControls hidden so only Plotly's chart zoom is visible
            setSupportZoom(true)
            builtInZoomControls                  = true
            displayZoomControls                  = false
        }

        webView.webChromeClient = WebChromeClient()

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, url: String) = false
        }

        webView.loadUrl("file:///android_asset/index.html")
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack()) {
            webView.goBack()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
}
