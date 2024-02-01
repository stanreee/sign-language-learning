import a from "../images/learnLetters/a.png"
import b from "../images/learnLetters/b.png"
import c from "../images/learnLetters/c.png"
import d from "../images/learnLetters/d.png"
import e from "../images/learnLetters/e.png"
import f from "../images/learnLetters/f.png"
import g from "../images/learnLetters/g.png"
import h from "../images/learnLetters/h.png"
import i from "../images/learnLetters/i.png"
import j from "../images/learnLetters/j.png"
import k from "../images/learnLetters/k.png"
import l from "../images/learnLetters/l.png"
import m from "../images/learnLetters/m.png"
import n from "../images/learnLetters/n.png"
import o from "../images/learnLetters/o.png"
import p from "../images/learnLetters/p.png"
import q from "../images/learnLetters/q.png"
import r from "../images/learnLetters/r.png"
import s from "../images/learnLetters/s.png"
import t from "../images/learnLetters/t.png"
import u from "../images/learnLetters/u.png"
import v from "../images/learnLetters/v.png"
import w from "../images/learnLetters/w.png"
import x from "../images/learnLetters/x.png"
import y from "../images/learnLetters/y.png"
import z from "../images/learnLetters/z.png"

import lettersResource from "../data/letters.json"
import VideoClip from "./VideoClip"

const DisplayLetters = () => {

    //DATA CALL
    const allLetters = lettersResource.allLetters;
    const letterDesc = lettersResource.lettersDesc;
    const letterVideos = lettersResource.letterVideos;

    return (
    <div className="Exercise-Page">
       <div className="Exercise-content">
        <h3>
            <div className = "letterAlign">
              <img src={a} alt= "a" /> 
              <h2>{allLetters[0]} </h2>
              <VideoClip letterVideoId= {letterVideos[0]}/>
            </div> 
            {letterDesc[0]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={b} alt="b" />
              <h2>{allLetters[1]} </h2>
              <VideoClip letterVideoId= {letterVideos[1]}/>
            </div>
            {letterDesc[1]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={c} alt="c" />
              <h2>{allLetters[2]} </h2>
              <VideoClip letterVideoId= {letterVideos[2]}/>
            </div>
            {letterDesc[2]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={d} alt="d" />
              <h2>{allLetters[3]} </h2>
              <VideoClip letterVideoId= {letterVideos[3]}/>
            </div>
            {letterDesc[3]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={e} alt="e" />
              <h2>{allLetters[4]} </h2>
              <VideoClip letterVideoId= {letterVideos[4]}/>
            </div>
            {letterDesc[4]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={f} alt="f" />
              <h2>{allLetters[5]} </h2>
              <VideoClip letterVideoId= {letterVideos[5]}/>
            </div>
            {letterDesc[5]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={g} alt="g" />
              <h2>{allLetters[6]} </h2>
              <VideoClip letterVideoId= {letterVideos[6]}/>
            </div>
            {letterDesc[6]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={h} alt="h" />
              <h2>{allLetters[7]} </h2>
              <VideoClip letterVideoId= {letterVideos[7]}/>
            </div>
            {letterDesc[7]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={i} alt="i" />
              <h2>{allLetters[8]} </h2>
              <VideoClip letterVideoId= {letterVideos[8]}/>
            </div>
            {letterDesc[8]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={j} alt="j" />
              <h2>{allLetters[9]} </h2>
              <VideoClip letterVideoId= {letterVideos[9]}/>
            </div>
            {letterDesc[9]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={k} alt="k" />
              <h2>{allLetters[10]} </h2>
              <VideoClip letterVideoId= {letterVideos[10]}/>
            </div>
            {letterDesc[10]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={l} alt="l" />
              <h2>{allLetters[11]} </h2>
              <VideoClip letterVideoId= {letterVideos[11]}/>
            </div>
            {letterDesc[11]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={m} alt="m" />
              <h2>{allLetters[12]} </h2>
              <VideoClip letterVideoId= {letterVideos[12]}/>
            </div>
            {letterDesc[12]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={n} alt="n" />
              <h2>{allLetters[13]} </h2>
              <VideoClip letterVideoId= {letterVideos[13]}/>
            </div>
            {letterDesc[13]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={o} alt="o" />
              <h2>{allLetters[14]} </h2>
              <VideoClip letterVideoId= {letterVideos[14]}/>
            </div>
            {letterDesc[14]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={p} alt="p" />
              <h2>{allLetters[15]} </h2>
              <VideoClip letterVideoId= {letterVideos[15]}/>
            </div>
            {letterDesc[15]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={q} alt="q" />
              <h2>{allLetters[16]} </h2>
              <VideoClip letterVideoId= {letterVideos[16]}/>
            </div>
            {letterDesc[16]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={r} alt="r" />
              <h2>{allLetters[17]} </h2>
              <VideoClip letterVideoId= {letterVideos[17]}/>
            </div>
            {letterDesc[17]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={s} alt="s" />
              <h2>{allLetters[18]} </h2>
              <VideoClip letterVideoId= {letterVideos[18]}/>
            </div>
            {letterDesc[18]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={t} alt="t" />
              <h2>{allLetters[19]} </h2>
              <VideoClip letterVideoId= {letterVideos[19]}/>
            </div>
            {letterDesc[19]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={u} alt="u" />
              <h2>{allLetters[20]} </h2>
              <VideoClip letterVideoId= {letterVideos[20]}/>
            </div>
            {letterDesc[20]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={v} alt="v" />
              <h2>{allLetters[21]} </h2>
              <VideoClip letterVideoId= {letterVideos[21]}/>
            </div>
            {letterDesc[21]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={w} alt="w" />
              <h2>{allLetters[22]} </h2>
              <VideoClip letterVideoId= {letterVideos[22]}/>
            </div>
            {letterDesc[22]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={x} alt="x" />
              <h2>{allLetters[23]} </h2>
              <VideoClip letterVideoId= {letterVideos[23]}/>
            </div>
            {letterDesc[23]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={y} alt="y" />
              <h2>{allLetters[24]} </h2>
              <VideoClip letterVideoId= {letterVideos[24]}/>
            </div>
            {letterDesc[24]}

            <div className="Section"> </div>
            <div className = "letterAlign">
              <img src={z} alt="z" />
              <h2>{allLetters[25]} </h2>
              <VideoClip letterVideoId= {letterVideos[25]}/>
            </div>
            {letterDesc[25]}
        </h3>
      </div>
    </div>
    );
}

export default DisplayLetters;
