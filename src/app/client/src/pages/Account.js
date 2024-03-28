import { useNavigate } from 'react-router-dom';
import asl from '../images/asl.png';

console.log(asl);

const Account = (props) => {

  const { loggedIn, userId, email, name, level } = props
  const navigate = useNavigate();

  console.log(props)
  
  const onLoginButtonClick = () => {
    if (loggedIn) {
        localStorage.removeItem("user")
        props.setLoggedIn(false)
    } else {
        navigate("/login")
    }
  }

  const onSignUpButtonClick = () => {
    if (loggedIn) {
        localStorage.removeItem("user")
        props.setLoggedIn(false)
    } else {
        navigate("/signup")
    }
  }

    return(

      <div className="mainContainer">
        {loggedIn ? 
          <div className="titleContainer">
            <h1>Welcome NAME!</h1> 
            <div>
            <h2>Profile Settings</h2>
            <a>You are signed in with {email}</a>
            <a>level: {level[2]}</a>
            <a>Total Score: {level[0]}/{level[1]}</a>
            </div>
          </div>
        
        : <div>
          Welcome to ASLingo
          <div>Login or Create an Account to Save Your Learning Progress!</div>
          </div>
        }
        {/* <div className={'titleContainer'}>
          {loggedIn ? <div>Welcome {name}!</div> : <div>Welcome to ASLingo</div>}
        </div> */}
        {/* {loggedIn ? 
          <div>
          <h2>Profile Settings</h2>
          <a>You are signed in with {email}</a>
          <a>level: {level[2]}</a>
          <a>Total Score: {level[0]}/{level[1]}</a>
          </div> : <div>Login or Create an Account to Save Your Learning Progress!</div>} */}

        <div className={'buttonContainer'}>
          <input
            className={'inputButton'}
            type="button"
            onClick={onLoginButtonClick}
            value={loggedIn ? 'Log out' : 'Log in'}
          />

          <input
            className={'inputButton'}
            type={loggedIn ? 'hidden' : 'button'}
            onClick={onSignUpButtonClick}
            value={loggedIn ? '' : 'Sign Up'}
          />
        </div>
      </div>

      );
}

export default Account;