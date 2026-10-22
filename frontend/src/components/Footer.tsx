export default function Footer() {
  return (
    <footer className="site-footer">
      <p className="site-footer__name">CardioLens AI</p>
      <p className="mx-auto mt-2 max-w-2xl">
        A research and learning project, not a certified medical device. Predictions come from a statistical model
        and must not be used to diagnose or treat anyone.
      </p>
      <p className="mt-2">
        © 2026 CardioLens AI.{' '}
        <a
          href="https://github.com/arwin-tech/cardiolens-ai"
          className="text-parchment underline decoration-gold/50 underline-offset-4 hover:text-cream"
          target="_blank"
          rel="noreferrer"
        >
          Source code on GitHub
        </a>
      </p>
    </footer>
  );
}
