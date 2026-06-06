let time = 30;

setInterval(() => {

    let timer = document.getElementById("timer");

    if(timer){

        timer.innerHTML = time;

        if(time > 0){
            time--;
        }

    }

},1000);