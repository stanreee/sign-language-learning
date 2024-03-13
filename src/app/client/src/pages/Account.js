import { useNavigate } from 'react-router-dom';
import asl from '../images/asl.png';

console.log(asl);

const Account = (props) => {

  const { loggedIn, email } = props
  const navigate = useNavigate();
  
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
        <div className={'titleContainer'}>
          <div>Welcome to ASLingo</div>
        </div>
        <p>Login or Create an Account to Save Your Learning Progress!</p>

        <div className={'buttonContainer'}>
          <input
            className={'inputButton'}
            type="button"
            onClick={onLoginButtonClick}
            value={loggedIn ? 'Log out' : 'Log in'}
          />
          {loggedIn ? <div>You are signed in with {email}</div> : <div />}

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