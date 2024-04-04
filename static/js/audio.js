
    function playAudio() {
      var input=document.getElementById("text").value;
      startAudio(input);
    }
    function startAudio(input) {
  // 检查浏览器是否支持Web Speech API
  if ('speechSynthesis' in window) {
      var successReceive=false;
    // 创建一个新的SpeechSynthesisUtterance对象
    let msg = new SpeechSynthesisUtterance(input);
    msg.volume = 5; //音量
    msg.rate = 1.2; //语速
    msg.text = input; //文字

    // 定义变量来跟踪重复请求的次数
    let retryCount = 0;

    // 定义函数来请求语音包列表
    function requestVoices() {
      let voices = window.speechSynthesis.getVoices();

      // 检查是否成功获取语音包列表
      if (voices.length > 0) {
        msg.voice = voices[62];
        speechSynthesis.speak(msg); //播放语音
        successReceive=true;
      } else {
        console.error('无法获取语音包列表');
        retryCount++;
        // 检查重复请求的次数
        if (retryCount < 5 && (successReceive==false)) {
          // 重新请求语音包列表
          setTimeout(requestVoices, 1000); // 1秒后再次请求
        } else {
          console.log('请求超时，停止重复请求');
        }
      }
    }

    // 请求语音包列表
    requestVoices();
  } else {
    console.error('浏览器不支持Web Speech API');
  }
}